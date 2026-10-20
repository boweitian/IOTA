"""Real browser → portal → GPU: two chat turns and a user-defined direction."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8768/')
    args=parser.parse_args();errors=[];results=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1440,height=1100))
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url+'#scenarios');frame=page.frame_locator('#scenariosFrame')
        frame.locator('#chatInput').wait_for()
        page.wait_for_function("!document.getElementById('scenariosFrame').contentDocument.getElementById('sendChat').disabled",timeout=120000)
        frame.locator('#agentInnerTab').click();frame.locator('#roleName').fill('澜舟')
        frame.locator('#customAxes').evaluate('(el)=>el.open=true')
        frame.locator('#customName').fill('轻松幽默')
        frame.locator('#customPositive').fill('用轻松幽默的语气回应，适当使用俏皮的比喻，但保持事实。')
        frame.locator('#customNegative').fill('用严肃正式的语气回应，不使用笑话或俏皮比喻。')
        with page.expect_response(lambda r:r.url.endswith('/api/studio/direction') and r.request.method=='POST',timeout=180000) as pending:
            frame.locator('#buildDirection').click()
        direction=pending.value.json();assert pending.value.status==200,direction
        assert len(direction['native_direction'])==4096
        cid=direction['axis']['id']
        frame.locator('#axis-'+cid).evaluate("el=>{el.value='0.5';el.dispatchEvent(new Event('input'))}")
        frame.locator('#agentChatTab').click()
        for index,question in enumerate(['你好，我叫小舟。请用一句话介绍你自己。','我刚才说自己叫什么？请只回答我的名字。']):
            frame.locator('#chatInput').fill(question)
            with page.expect_response(lambda r:r.url.endswith('/api/studio/chat') and r.request.method=='POST',timeout=180000) as pending:
                frame.locator('#sendChat').click()
            response=pending.value;result=response.json();assert response.status==200,result
            frame.locator('.chat-message.assistant').nth(index).wait_for()
            assert frame.locator('.chat-message.assistant .chat-text').nth(index).inner_text()==result['message']['content']
            assert result['profile']['role']['name']=='澜舟'
            assert result['profile']['coefficients'][cid]==.5
            assert result['context_messages']==index*2+1
            assert result['provenance']['model']['device']!='CPU'
            intervention=result['intervention']
            if intervention.get('regularization')=='direction_projection':
                assert max(map(abs,intervention['projections'].values()),default=0.)<=intervention['projection_limit']+1e-5
            elif intervention.get('regularization','norm')=='norm':
                assert intervention['norm']<=intervention['cap']+1e-5
            else:
                assert intervention['cap_factor']==1.
            results.append(dict(request=response.request.post_data_json,result=result))
        assert '澜舟' in results[0]['result']['message']['content'],results[0]['result']['message']
        assert '小舟' in results[1]['result']['message']['content'],results[1]['result']['message']
        page.reload();frame.locator('.chat-message.assistant').nth(1).wait_for()
        assert frame.locator('#agentName').inner_text()=='澜舟'
        page.evaluate('scrollTo(0,0)');page.screenshot(path=str(ROOT/'runtime/agent-live-desktop.png'),full_page=True)
        page.set_viewport_size(dict(width=390,height=844));page.evaluate('scrollTo(0,0)')
        assert page.evaluate('document.documentElement.scrollWidth')<=390
        page.screenshot(path=str(ROOT/'runtime/agent-live-mobile.png'),full_page=True)
        assert not errors,errors
        result=dict(passed=True,api_mode='real GPU inference',url=args.url+'#scenarios',
            checks=['user-defined direction built on Qwen3.5-9B','named agent and same custom geometry in both turns',
                    'native conversation history and second-turn name recall','exact model outputs displayed',
                    'browser reload preserves identity and conversation'],direction=direction,turns=results,page_errors=errors)
        (ROOT/'runtime/agent-live-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(passed=True,checks=result['checks'],replies=[r['result']['message']['content'] for r in results]),ensure_ascii=False,indent=2))
        browser.close()


if __name__=='__main__':main()
