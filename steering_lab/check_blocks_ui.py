"""Block editor workflows with explicit API stubs; no fabricated model evidence."""
import json
import threading
import numpy as np
from studio import mixed_shift
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kw):super().__init__(*args,directory=str(ROOT/'dist'),**kw)
        def log_message(self,*_):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f'http://127.0.0.1:{server.server_port}/';errors=[];requests=[]
    data=json.loads((ROOT/'dist/studio-evidence.json').read_text())
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport=dict(width=1540,height=1100))
        def route_api(route):
            path=route.request.url.split('/api/',1)[1]
            if path=='status':body=dict(online=True,studio=dict(supports_blocks=True,maximum_strength=2.))
            elif path=='latest':body=dict(latest=None)
            elif path=='studio/chat':
                request=route.request.post_data_json;requests.append(request)
                vectors={data['faith_preset']['axis']['id']:np.array(data['faith_preset']['native_direction']),
                    'custom-block-test':np.array(data['atlas']['concepts'][0]['native_direction'])}
                _,intervention=mixed_shift(request['profile'],np.array([c['native_direction'] for c in data['atlas']['concepts']]),
                    data['atlas']['scale']*2,vectors,regularization=data['strength_regularization'])
                body=dict(message=dict(role='assistant',content='明确的 UI 测试回复，不是真实模型输出。'),
                    profile=request['profile'],profile_sha256='ui-test',agent_name=request['profile']['role']['name'],
                    generation=dict(truncated=False),intervention=intervention)
            elif path=='studio/direction':
                axis=route.request.post_data_json['axis'];axis['id']='custom-block-test'
                c=data['atlas']['concepts'][0]
                body=dict(axis=axis,native_direction=c['native_direction'],direction=c['direction'],points=[],
                    diagnostics=dict(split_half=.9,nearest=[]),provenance=dict(protocol_sha256=data['protocol_sha256']))
            else:route.fulfill(status=404,json=dict(error='unknown test route'));return
            route.fulfill(json=body)
        context.route('**/api/**',route_api)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url);frame=page.frame_locator('#scenariosFrame')
        frame.locator('#agentInnerTab').click();frame.locator('#blocksCanvas').wait_for()
        assert frame.locator('#legacyGeometry').is_hidden()
        assert frame.locator('.intervention-block').count()==0
        assert frame.locator('#addBlock').count()==1
        assert page.evaluate("Object.values(document.getElementById('scenariosFrame').contentWindow.eval('profile.coefficients')).every(v=>v===0)")
        frame.locator('#roleName').fill('星澜')
        frame.locator('#addBlock').click();assert frame.locator('#blockInspector').is_visible()
        frame.locator('#blockDirection').select_option('warmth')
        assert frame.locator('#blockStrength').get_attribute('max')=='2'
        assert '1× 为标准干预' in frame.locator('.block-risk').inner_text()
        frame.locator('#applyBlock').click()
        assert frame.locator('.intervention-block').count()==1
        assert frame.locator('#addBlock').count()==1
        frame.locator('#addBlock').click();frame.locator('#blockDirection').select_option('brevity')
        frame.locator('#blockStrength').evaluate("el=>{el.value='0.5';el.dispatchEvent(new Event('input'))}")
        frame.locator('#applyBlock').click();assert frame.locator('.intervention-block').count()==2
        frame.locator('[data-direction="warmth"] .block-edit').click()
        frame.locator('#blockStrength').evaluate("el=>{el.value='2';el.dispatchEvent(new Event('input'))}")
        frame.locator('#applyBlock').click()
        assert frame.locator('#blocksHighWarning').is_visible()
        assert frame.locator('[data-direction="warmth"] .block-dose').inner_text()=='2.00×'
        frame.locator('#deleteBlock').click()
        assert frame.locator('.intervention-block').count()==1
        assert frame.locator('[data-direction="brevity"]').is_visible()
        page.reload();frame.locator('#agentInnerTab').click()
        assert frame.locator('.intervention-block').count()==1
        assert frame.locator('#roleName').input_value()=='星澜'
        frame.locator('#customAxes').evaluate('el=>el.open=true')
        frame.locator('#customName').fill('幽默积木');frame.locator('#customPositive').fill('轻松幽默。');frame.locator('#customNegative').fill('严肃正式。')
        frame.locator('#buildDirection').click();frame.locator('#axis-custom-block-test').wait_for()
        assert frame.locator('#blockDirection').input_value()=='custom-block-test'
        frame.locator('#blockStrength').evaluate("el=>{el.value='1.25';el.dispatchEvent(new Event('input'))}")
        frame.locator('#applyBlock').click();assert frame.locator('.intervention-block').count()==2
        frame.locator('#verifyBlock').click();frame.locator('#chatInput').fill('这是一条界面测试消息。')
        frame.locator('#sendChat').click();frame.locator('.chat-message.assistant').wait_for()
        assert requests[-1]['profile']['coefficients']['custom-block-test']==1.25
        frame.locator('#agentInnerTab').click();frame.locator('[data-direction="custom-block-test"] .block-edit').click()
        assert frame.locator('#confirmBlock').is_visible()
        frame.locator('#confirmBlock').click()
        assert '你已确认有效' in frame.locator('[data-direction="custom-block-test"]').inner_text()
        frame.locator('#legacyDirectionControls').evaluate('el=>el.open=true')
        assert frame.locator('#faithPreset, #restoreFaithExperiment, #undoFaithExperiment').count()==0
        faith=data['faith_preset']['axis']['id']
        frame.locator('#clearBlocks').click();frame.locator('#addBlock').click()
        frame.locator('#blockDirection').select_option(faith)
        frame.locator('#blockStrength').evaluate("el=>{el.value='2';el.dispatchEvent(new Event('input'))}")
        frame.locator('#applyBlock').click()
        assert frame.locator('.intervention-block').count()==1
        assert frame.locator('[data-direction="'+faith+'"] .block-dose').inner_text()=='2.00×'
        assert frame.locator('#blocksHighWarning').is_visible()
        assert frame.locator('#roleName').input_value()=='星澜'
        frame.locator('#verifyBlock').click();frame.locator('#newConversation').click()
        frame.locator('#chatInput').fill('你相信上帝吗？');frame.locator('#sendChat').click()
        frame.locator('.chat-message.assistant').wait_for()
        last=requests[-1]['profile']
        assert last['coefficients'][faith]==2.
        assert all(v==0 for k,v in last['coefficients'].items() if k!=faith)
        assert last['custom_axes'][0]['source']=='identity-response-20261017-v1'
        frame.locator('#agentInnerTab').click()
        with page.expect_download() as download:frame.locator('#exportProfile').click()
        exported_bytes=Path(download.value.path()).read_bytes();exported=json.loads(exported_bytes)
        assert exported['blocks']['order']==[faith]
        assert exported['profile']['coefficients'][faith]==2.
        frame.locator('#agentInnerTab').click();frame.locator('#clearBlocks').click()
        frame.locator('#profileFile').set_input_files(dict(name='faith.json',mimeType='application/json',buffer=exported_bytes))
        page.wait_for_function("document.getElementById('scenariosFrame').contentDocument.getElementById('saveStatus').textContent.includes('已导入')")
        assert frame.locator('.intervention-block').count()==1
        assert frame.locator('.block-dose').inner_text()=='2.00×'
        frame.locator('#agentInnerTab').click()
        assert frame.locator('#roleName').input_value()=='星澜'
        assert frame.locator('.intervention-block').count()==1
        assert frame.locator('#blockInspector').is_hidden()
        page.evaluate('scrollTo(0,0)');page.screenshot(path=str(ROOT/'runtime/blocks-desktop.png'),full_page=True)
        page.set_viewport_size(dict(width=390,height=844))
        assert page.evaluate('document.documentElement.scrollWidth')<=390
        assert page.evaluate("document.getElementById('scenariosFrame').contentDocument.documentElement.scrollWidth")<=390
        frame.locator('#addBlock').click()
        assert frame.locator('#blockInspector').is_visible()
        assert page.evaluate("document.getElementById('scenariosFrame').contentDocument.documentElement.scrollWidth")<=390
        page.evaluate('scrollTo(0,0)');page.screenshot(path=str(ROOT/'runtime/blocks-mobile.png'),full_page=True)
        frame.locator('#clearBlocks').click();assert frame.locator('.intervention-block').count()==0
        page.reload();frame.locator('#agentInnerTab').click();assert frame.locator('.intervention-block').count()==0
        assert frame.locator('#addBlock').count()==1
        assert not errors,errors
        result=dict(passed=True,api_mode='explicit stubs; real-model execution tested separately',page_errors=errors,
            checks=['fresh canvas empty with one dashed starting node','recursive add, edit to 2x, delete preserving later nodes',
                    'custom directions and persistence','actual profile payload at 1.25x and experimental faith at 2x',
                    'inference and human confirmation distinguished','identity direction uses normal block editing; original exported file import','390px mobile and desktop layouts','clear and reload remain empty'])
        (ROOT/'runtime/blocks-ui-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        legacy=context.browser.new_context(viewport=dict(width=1440,height=1000))
        legacy.route('**/api/**',route_api)
        legacy_binding={k:data['atlas']['model'][k] for k in ['model','revision','layer_index']}
        legacy_binding['protocol_sha256']=data['protocol_sha256']
        seed=dict(binding=legacy_binding,profile=data['presets'][1]['profile'],messages=[])
        legacy.add_init_script("localStorage.setItem('iota-agent-state-v1',"+json.dumps(json.dumps(seed,ensure_ascii=False))+");")
        old=legacy.new_page();old.on('pageerror',lambda e:errors.append(str(e)));old.goto(url+'?view=inner#scenarios')
        oldframe=old.frame_locator('#scenariosFrame');oldframe.locator('#blocksCanvas').wait_for(state='visible')
        assert oldframe.locator('.intervention-block').count()==sum(v!=0 for v in seed['profile']['coefficients'].values())
        assert oldframe.locator('#blocksMigration').is_visible()
        assert oldframe.locator('#axis-courage').input_value()==str(seed['profile']['coefficients']['courage'])
        result['checks'].append('existing active profiles migrate to blocks without losing coefficients')
        assert not errors,errors
        (ROOT/'runtime/blocks-ui-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2));legacy.close();browser.close()
    server.shutdown()


if __name__=='__main__':main()
