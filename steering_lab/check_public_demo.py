"""Real HTTPS → public gateway → existing portal → Qwen GPU and browser UI."""
import json
import base64
import urllib.error
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent;RUNTIME=ROOT/'runtime/public-demo'


def main():
    meta=json.loads((RUNTIME/'active.json').read_text());url=meta['url']
    protected=meta.get('password_protected',True)
    access=json.loads((RUNTIME/'access.json').read_text()) if protected else None
    authorization='Basic '+base64.b64encode((access['username']+':'+access['password']).encode()).decode() if access else None
    def request(path,data=None,headers=None,auth=True):
        h=dict(headers or {})
        if authorization and auth:h['Authorization']=authorization
        q=urllib.request.Request(url+path,data=data,headers=h)
        try:
            with urllib.request.urlopen(q,timeout=40) as r:return r.status,r.read()
        except urllib.error.HTTPError as error:return error.code,error.read()
    for path in ['/','/api/status','/api/latest']:
        status,_=request(path,auth=False);assert status==(401 if protected else 200),(path,status)
    assert request('/api/status',headers={'Authorization':'Basic aW90YTppbnZhbGlk'},auth=False)[0]==(401 if protected else 200)
    status,body=request('/api/status');state=json.loads(body);assert status==200 and state['online']
    assert state['studio']['model']['model']=='Qwen/Qwen3.5-9B'
    assert json.loads(request('/api/latest')[1])==dict(latest=None)
    for path in ['/assets/','/runtime/public-demo/access.json','/api/unknown']:
        status,_=request(path);assert status==404,(path,status)
    # The edge can reject encoded traversal before it reaches our gateway.
    assert request('/%2e%2e/interactive.py')[0] in [400,404]
    assert request('/api/studio/chat',data=b'{}',headers={'Content-Type':'application/json','Origin':'https://unrelated.example'})[0]==403
    assert request('/api/studio/chat',data=b'{}',headers={'Content-Type':'text/plain'})[0]==415
    errors=[];turns=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport=dict(width=1440,height=1100),**({'http_credentials':access} if access else {}))
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url+'/#scenarios');frame=page.frame_locator('#scenariosFrame');frame.locator('#chatInput').wait_for(timeout=60000)
        page.wait_for_function("!document.getElementById('scenariosFrame').contentDocument.getElementById('sendChat').disabled",timeout=120000)
        for index,question in enumerate(['你好，我叫公网小舟。请用一句话介绍你自己。','我刚才告诉你的名字是什么？请只回答名字。']):
            frame.locator('#chatInput').fill(question)
            with page.expect_response(lambda r:r.url.endswith('/api/studio/chat') and r.request.method=='POST',timeout=180000) as pending:
                frame.locator('#sendChat').click()
            response=pending.value;result=response.json();assert response.status==200,result
            assert result['provenance']['model']['model']=='Qwen/Qwen3.5-9B'
            assert result['provenance']['model']['device']!='CPU'
            assert result['context_messages']==index*2+1
            displayed=frame.locator('.chat-message.assistant .chat-text').nth(index)
            displayed.wait_for();assert displayed.inner_text()==result['message']['content']
            turns.append(result)
        assert '公网小舟' in turns[1]['message']['content'],turns[1]['message']
        page.reload();frame.locator('.chat-message.assistant').nth(1).wait_for()
        page.locator('#capabilitiesTab').click();nav=page.frame_locator('#capabilitiesFrame')
        evidence=json.loads((ROOT/'dist/workspace-evidence.json').read_text())
        baseline=nav.locator('#scores .big').first if evidence.get('output_protocol')=='analysis_then_答案' else nav.locator('#sweepBase')
        baseline.wait_for()
        expected=evidence['sweep_metrics']['baseline']['accuracy']
        assert baseline.inner_text()==f'{expected:.1%}'
        page.locator('#scenariosTab').click();page.evaluate('scrollTo(0,0)')
        page.screenshot(path=str(RUNTIME/'public-desktop.png'),full_page=True)
        page.set_viewport_size(dict(width=390,height=844));page.evaluate('scrollTo(0,0)')
        assert page.evaluate('document.documentElement.scrollWidth')<=390
        page.screenshot(path=str(RUNTIME/'public-mobile.png'),full_page=True)
        assert not errors,errors
        browser.close()
    result=dict(passed=True,url=url,expires_at=meta['expires_at'],model=turns[0]['provenance']['model'],
                checks=['HTTPS reachable','visitor authentication required' if protected else 'pages and visitor APIs work without credentials',
                        'private portal latest result isolated','directory listing and traversal denied','cross-origin writes denied',
                        'two real GPU chat turns over the public browser','model output and conversation history preserved',
                        'Navigator scan data available','390px mobile layout'],
                replies=[r['message']['content'] for r in turns],page_errors=errors)
    (RUNTIME/'check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
