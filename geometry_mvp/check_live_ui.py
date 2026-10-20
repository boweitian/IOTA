"""Real browser + real local inference, including knob regeneration."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8766');args=p.parse_args()
    root=Path(__file__).resolve().parent
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1440,height=1050))
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        replies=[]
        def capture(res):
            if res.url.endswith('/api/compare'):replies.append(res.json())
        page.on('response',capture)
        page.goto(args.url);page.wait_for_function('state.live')
        prompt=page.locator('#prompt').inner_text()
        page.locator('.custom summary').click();page.locator('#custom-prompt').fill(prompt)
        page.locator('#generate').click()
        page.wait_for_function('state.custom && !state.busy',timeout=90000)
        assert page.locator('#prompt').inner_text()==prompt
        assert replies[-1]['baseline_strength']==-2 and replies[-1]['strength']==2
        print('live baseline:',page.locator('#baseline').inner_text(),'steered:',page.locator('#steered').inner_text(),flush=True)
        page.locator('#strength').fill('1')
        page.wait_for_function('state.strength===1 && state.custom && !state.custom.pending && !state.busy',timeout=90000)
        assert page.locator('#prompt').inner_text()==prompt
        assert replies[-1]['strength']==1 and len(replies)==2
        assert not errors,errors
        (root/'live_browser_check.json').write_text(json.dumps(dict(prompt=prompt,calls=replies,browser_errors=errors),ensure_ascii=False,indent=2))
        page.screenshot(path=str(root/'preview_live.png'),full_page=True)
        print('PASS: live status, actual custom generations, pinned source, knob regeneration, preserved prompt; no JS errors.')
        browser.close()
if __name__=='__main__':main()
