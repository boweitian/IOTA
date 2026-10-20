"""Browser regressions with explicit API stubs; real inference is checked separately."""
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT/'dist'),**kw)
        def log_message(self,*_):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f'http://127.0.0.1:{server.server_port}/'
    requests=[];errors=[];online=True;fail=False
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport=dict(width=1440,height=1100))
        def handle(route):
            nonlocal fail
            path=route.request.url.split('/api/',1)[1]
            if path=='status':body=dict(online=online)
            elif path=='latest':body=dict(latest=None)
            elif path=='studio/chat':
                request=route.request.post_data_json;requests.append(request)
                if fail:
                    fail=False;route.fulfill(status=503,json=dict(error='test service unavailable'));return
                body=dict(message=dict(role='assistant',content='测试回复 '+str(len(requests))+' <b>literal</b>'),
                    profile=request['profile'],agent_name=request['profile']['role']['name'],profile_sha256='test-profile',
                    intervention=dict(norm=.1,cap_factor=.8),generation=dict(truncated=False))
            elif path=='studio/direction':
                axis=route.request.post_data_json['axis'];axis['id']='custom-ui-test'
                data=json.loads((ROOT/'dist/studio-evidence.json').read_text())
                c=data['atlas']['concepts'][0]
                body=dict(axis=axis,native_direction=c['native_direction'],direction=c['direction'],points=[],
                    diagnostics=dict(split_half=.9,nearest=[]),provenance=dict(protocol_sha256=data['protocol_sha256']))
            else:route.fulfill(status=404,json=dict(error='unknown test route'));return
            route.fulfill(json=body)
        context.route('**/api/**',handle)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url);frame=page.frame_locator('#scenariosFrame')
        frame.locator('#chatInput').wait_for()
        assert frame.locator('#agentChatPanel').is_visible()
        frame.locator('#agentInnerTab').click()
        frame.locator('#roleName').fill('星澜')
        frame.locator('#legacyDirectionControls').evaluate('el=>el.open=true');frame.locator('[data-preset="quiet"]').click()
        assert frame.locator('#roleName').input_value()=='星澜'
        assert '星澜' in frame.locator('#roleIdentity').input_value()
        assert '伊澜' not in frame.locator('#roleIdentity').input_value()
        frame.locator('#axis-warmth').evaluate("el=>{el.value='0.65';el.dispatchEvent(new Event('input'))}")
        frame.locator('#agentChatTab').click()
        assert frame.locator('#chatAgentName').inner_text()=='星澜'
        frame.locator('#chatInput').fill('你好，我叫小舟。');frame.locator('#sendChat').click()
        frame.locator('.chat-message.assistant').wait_for()
        assert requests[-1]['profile']['coefficients']['warmth']==.65
        assert requests[-1]['messages']==[dict(role='user',content='你好，我叫小舟。')]
        assert frame.locator('.chat-text b').count()==0
        frame.locator('#agentInnerTab').click();frame.locator('#axis-warmth').evaluate("el=>{el.value='-0.35';el.dispatchEvent(new Event('input'))}")
        frame.locator('#agentChatTab').click();frame.locator('#chatInput').fill('你还记得我的名字吗？')
        frame.locator('#chatInput').press('Enter')
        frame.locator('.chat-message.assistant').nth(1).wait_for()
        assert requests[-1]['profile']['coefficients']['warmth']==-.35
        assert [m['role'] for m in requests[-1]['messages']]==['user','assistant','user']
        assert requests[-1]['messages'][0]['content']=='你好，我叫小舟。'
        page.reload();frame.locator('.chat-message.assistant').nth(1).wait_for()
        assert frame.locator('#agentName').inner_text()=='星澜'
        frame.locator('#agentInnerTab').click()
        assert frame.locator('#axis-warmth').input_value()=='-0.35'
        frame.locator('#customAxes').evaluate('(el)=>el.open=true')
        frame.locator('#customName').fill('幽默');frame.locator('#customPositive').fill('轻松幽默。');frame.locator('#customNegative').fill('严肃正式。')
        frame.locator('#buildDirection').click();frame.locator('#axis-custom-ui-test').wait_for()
        frame.locator('#axis-custom-ui-test').evaluate("el=>{el.value='0.5';el.dispatchEvent(new Event('input'))}")
        frame.locator('#agentChatTab').click();frame.locator('#chatInput').fill('讲个笑话。');frame.locator('#sendChat').click()
        frame.locator('.chat-message.assistant').nth(2).wait_for()
        assert requests[-1]['profile']['custom_axes'][0]['name']=='幽默'
        assert requests[-1]['profile']['coefficients']['custom-ui-test']==.5
        assert '幽默 +0.40' in frame.locator('.chat-message.assistant').nth(2).text_content()
        assert '共同乘以 0.80' in frame.locator('.chat-message.assistant').nth(2).text_content()
        fail=True;frame.locator('#chatInput').fill('失败后保留这句话');frame.locator('#sendChat').click()
        page.wait_for_function("document.getElementById('scenariosFrame').contentDocument.getElementById('chatStatus').textContent.includes('发送未完成')")
        assert frame.locator('#chatInput').input_value()=='失败后保留这句话'
        assert frame.locator('.chat-message.user').count()==3
        frame.locator('#sendChat').click();frame.locator('.chat-message.assistant').nth(3).wait_for()
        frame.locator('#newConversation').click()
        assert frame.locator('.chat-message').count()==0
        assert frame.locator('#agentName').inner_text()=='星澜'
        page.set_viewport_size(dict(width=390,height=844))
        for tab in ['agentChatTab','agentInnerTab']:
            frame.locator('#'+tab).click();page.wait_for_timeout(100)
            assert page.evaluate('document.documentElement.scrollWidth')<=390
            assert page.evaluate("document.getElementById('scenariosFrame').contentDocument.documentElement.scrollWidth")<=390
        frame.locator('#agentChatTab').click();page.screenshot(path=str(ROOT/'runtime/agent-mobile.png'),full_page=True)
        page.set_viewport_size(dict(width=1440,height=1100))
        page.screenshot(path=str(ROOT/'runtime/agent-desktop.png'),full_page=True)
        frame.locator('#agentInnerTab').click();page.wait_for_timeout(100)
        page.screenshot(path=str(ROOT/'runtime/agent-inner.png'),full_page=True)
        online=False;page.reload();frame.locator('#chatInput').wait_for()
        assert frame.locator('#sendChat').is_disabled()
        assert not errors,errors
        result=dict(passed=True,api_mode='explicit stubs, not real-model evidence',checks=[
            'same named agent across both interfaces and behavior presets',
            'native multi-turn payload, changed built-in and custom shifts',
            'draft and conversation reload persistence',
            'failed send preserves draft and retries without duplicate turns',
            'new conversation preserves identity, offline prevents fake replies',
            '390px mobile and desktop layouts, literal text rendering'],page_errors=errors)
        (ROOT/'runtime/agent-ui-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        browser.close()
    server.shutdown()


if __name__=='__main__':main()
