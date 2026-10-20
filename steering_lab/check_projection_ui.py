"""Independent geometry checks, browser/Python parity, optional real GPU validation."""
import argparse
import json
import math
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from playwright.sync_api import sync_playwright
from studio import AXES, mixed_shift
from test_projection import CASES, case_profile

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');args=parser.parse_args()
    server=None
    if args.live:
        url='http://127.0.0.1:8768/'
        for attempt in range(40):
            status=json.load(urlopen(url+'api/status',timeout=5))
            if status.get('job_state')=='RUNNING' and status.get('online') and status.get('state')=='ready' and status.get('studio',{}).get('strength_regularization')=='direction_projection':break
            if attempt%6==0:print('Waiting for projection GPU worker:',status.get('job_state'),status.get('state'),flush=True)
            time.sleep(5)
        else:raise RuntimeError('Projection GPU worker not ready')
    else:
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT/'dist'),**kw)
            def log_message(self,*a):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        url=f'http://127.0.0.1:{server.server_port}/'
    errors=[];records=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page()
            if not args.live:page.route('**/api/**',lambda route:route.fulfill(json=dict(online=False,latest=None)))
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url+'?view=inner#scenarios');frame=page.frame_locator('#scenariosFrame')
            frame.locator('#blocksCanvas').wait_for()
            for case in CASES:
                profile,directions=case_profile(case)
                _,expected=mixed_shift(profile,directions,1.,regularization='direction_projection')
                actual=frame.locator('body').evaluate('''(_,test)=>IOTARegularization.combine(
                  test.vectors.map((v,i)=>({id:String(i),native_direction:v,direction:[0,0,0]})),
                  Object.fromEntries(test.coefficients.map((v,i)=>[String(i),v])),1,'direction_projection')''',case)
                assert math.isclose(actual['cap_factor'],case['factor'],abs_tol=1e-12),case['name']
                assert math.isclose(actual['norm'],expected['norm'],abs_tol=1e-12),case['name']
                assert actual['max_projection']<=expected['projection_limit']+1e-12,case['name']
                assert math.isclose(actual['max_raw_projection'],expected['max_raw_projection'],abs_tol=1e-12),case['name']
            # Built-in direction geometry: first pair is positively correlated.
            data=json.loads((ROOT/'dist/studio-evidence.json').read_text())
            directions=np.array([c['native_direction'] for c in data['atlas']['concepts']])
            for second in [1.,-1.]:
                setup=frame.locator('body').evaluate('''(_,second)=>{
                  for(const id of Object.keys(profile.coefficients))profile.coefficients[id]=0;
                  profile.coefficients[C[0].id]=1;profile.coefficients[C[1].id]=second;
                  syncInputs();refresh();return {profile:normalize(profile),mix:mixture()};
                }''',second)
                _,expected=mixed_shift(setup['profile'],directions,data['atlas']['scale'],regularization='direction_projection')
                assert math.isclose(setup['mix']['cap_factor'],expected['cap_factor'],rel_tol=1e-9)
                assert '最大方向投影' in frame.locator('#blocksSummary').inner_text()
                assert '上限 1.00×' in frame.locator('#normNote').inner_text()
                if args.live:
                    frame.locator('#agentChatTab').click();frame.locator('#newConversation').click()
                    frame.locator('#chatInput').fill('17 加 25 是多少？请只回答数字。')
                    with page.expect_response(lambda r:r.url.endswith('/api/studio/chat') and r.request.method=='POST',timeout=180000) as pending:frame.locator('#sendChat').click()
                    response=pending.value;result=response.json();assert response.status==200,result
                    info=result['intervention'];assert info['regularization']=='direction_projection'
                    assert math.isclose(info['cap_factor'],expected['cap_factor'],rel_tol=1e-6)
                    assert math.isclose(info['norm'],setup['mix']['norm'],rel_tol=1e-6)
                    actual_vector=np.array(setup['mix']['native_direction'])
                    projections=directions[:2]@actual_vector/np.einsum('ij,ij->i',directions[:2],directions[:2])
                    assert np.max(np.abs(projections))<=1.+1e-6
                    if second>0:
                        assert info['cap_factor']<1
                        assert math.isclose(info['max_projection'],1.,rel_tol=1e-6)
                    else:assert info['cap_factor']==1.
                    frame.locator('.chat-message.assistant').wait_for()
                    assert '方向投影上限' in frame.locator('.chat-message.assistant').text_content()
                    assert '达到总强度上限' not in frame.locator('.chat-message.assistant').text_content()
                    records.append(dict(second_coefficient=second,intervention=info,reply=result['message']['content']))
                    frame.locator('#agentInnerTab').click()
            if args.live:
                page.reload();frame.locator('.chat-message.assistant').wait_for(state='attached')
                assert '方向投影上限' in frame.locator('.chat-message.assistant').text_content()
            assert not errors,errors
            report=dict(passed=True,mode='real GPU' if args.live else 'browser geometry; no model inference',
                synthetic_cases=[case['name'] for case in CASES],turns=records,page_errors=errors)
            target=ROOT/'runtime'/('projection-live-check.json' if args.live else 'projection-ui-check.json')
            target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
            browser.close()
    finally:
        if server:server.shutdown();server.server_close()

if __name__=='__main__':main()
