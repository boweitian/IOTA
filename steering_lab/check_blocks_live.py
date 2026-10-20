"""Real canvas → actual 2x profile → full-position vector injection → GPU reply."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    errors=[];turns=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1540,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://127.0.0.1:8768/?view=inner#scenarios')
        frame=page.frame_locator('#scenariosFrame');frame.locator('#blocksCanvas').wait_for(state='visible')
        assert frame.locator('.intervention-block').count()==0
        frame.locator('#interventionBuilder').screenshot(path=str(ROOT/'runtime/blocks-live-empty.png'))
        original=json.loads((ROOT/'dist/studio-evidence.json').read_text())['faith_preset']['original_role']
        frame.locator('#roleIdentity').fill(original['identity'])
        frame.locator('#roleFacts').fill('\n'.join(original['facts']))
        frame.locator('#roleBoundaries').fill(original['boundaries'])
        frame.locator('#addBlock').click()
        frame.locator('#blockDirection').select_option('experiment-faith-20261017-v1')
        frame.locator('#blockStrength').evaluate("el=>{el.value='2';el.dispatchEvent(new Event('input'))}")
        frame.locator('#applyBlock').click()
        assert frame.locator('.intervention-block').count()==1
        assert frame.locator('.block-dose').inner_text()=='2.00×'
        frame.locator('#verifyBlock').click()
        page.wait_for_function("!document.getElementById('scenariosFrame').contentDocument.getElementById('sendChat').disabled",timeout=120000)
        question='你相信上帝吗？请说说你自己的想法。'
        frame.locator('#chatInput').fill(question)
        with page.expect_response(lambda r:r.url.endswith('/api/studio/chat') and r.request.method=='POST',timeout=180000) as pending:
            frame.locator('#sendChat').click()
        response=pending.value;result=response.json();assert response.status==200,result
        frame.locator('.chat-message.assistant').wait_for()
        assert frame.locator('.chat-message.assistant .chat-text').inner_text()==result['message']['content']
        assert result['intervention']['strength_multiplier']==2.
        assert result['intervention']['full_position_axes']==['experiment-faith-20261017-v1']
        assert abs(result['intervention']['norm']-2*result['intervention']['standard_cap'])<1e-4
        assert result['provenance']['model']['device']!='CPU'
        turns.append(dict(request=response.request.post_data_json,result=result))
        frame.locator('#agentInnerTab').click()
        assert '已试聊' in frame.locator('.block-phase').inner_text()
        assert '你已确认有效' not in frame.locator('.block-phase').inner_text()
        frame.locator('#closeBlockEditor').click()
        frame.locator('#interventionBuilder').screenshot(path=str(ROOT/'runtime/blocks-live-faith.png'))
        frame.locator('#addBlock').click();frame.locator('#blockDirection').select_option('brevity')
        frame.locator('#blockStrength').evaluate("el=>{el.value='0.5';el.dispatchEvent(new Event('input'))}")
        frame.locator('#applyBlock').click()
        assert frame.locator('.intervention-block').count()==2
        frame.locator('[data-direction="experiment-faith-20261017-v1"] .block-actions button').filter(has_text='删除').click()
        assert frame.locator('.intervention-block').count()==1
        frame.locator('[data-direction="brevity"] .block-edit').click()
        frame.locator('#blockStrength').evaluate("el=>{el.value='2';el.dispatchEvent(new Event('input'))}")
        frame.locator('#applyBlock').click();frame.locator('#verifyBlock').click();frame.locator('#newConversation').click()
        frame.locator('#chatInput').fill('17 加 25 是多少？请只回答数字。')
        with page.expect_response(lambda r:r.url.endswith('/api/studio/chat') and r.request.method=='POST',timeout=180000) as pending:
            frame.locator('#sendChat').click()
        response=pending.value;ordinary=response.json();assert response.status==200,ordinary
        assert ordinary['intervention']['strength_multiplier']==2.
        assert ordinary['intervention']['full_position_axes']==[]
        assert '42' in ordinary['message']['content']
        turns.append(dict(request=response.request.post_data_json,result=ordinary))
        frame.locator('#agentInnerTab').click();frame.locator('#closeBlockEditor').click()
        page.set_viewport_size(dict(width=390,height=844));page.evaluate('scrollTo(0,0)')
        assert page.evaluate('document.documentElement.scrollWidth')<=390
        assert page.evaluate("document.getElementById('scenariosFrame').contentDocument.documentElement.scrollWidth")<=390
        frame.locator('#interventionBuilder').screenshot(path=str(ROOT/'runtime/blocks-live-mobile.png'))
        page.reload();frame.locator('#blocksCanvas').wait_for(state='visible')
        assert frame.locator('.intervention-block').count()==1
        assert frame.locator('.block-dose').inner_text()=='2.00×'
        assert not errors,errors
        report=dict(passed=True,api_mode='real Qwen3.5-9B GPU',checks=['empty default canvas and direct inner-page link',
            'identity block applies actual 2x norm and full-position vector','exact GPU text displayed, node marked tested only',
            'deleting faith block removes full-position injection','ordinary direction also supports real 2x norm','reload and mobile layout'],
            turns=turns,page_errors=errors,manual_behavior_confirmations=0)
        (ROOT/'runtime/blocks-live-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(passed=True,checks=report['checks'],faith_reply=result['message']['content'],ordinary_reply=ordinary['message']['content']),ensure_ascii=False,indent=2))
        browser.close()


if __name__=='__main__':main()
