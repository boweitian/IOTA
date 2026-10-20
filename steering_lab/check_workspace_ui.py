"""The served two-panel workspace, retained state, old evidence and actual GPU API."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8768/');p.add_argument('--skip-live',action='store_true');args=p.parse_args()
    evidence=json.loads((ROOT/'dist/studio-evidence.json').read_text());errors=[];checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(args.url,wait_until='load').status==200
        cap=page.frame_locator('#capabilitiesFrame');cap.locator('#nodeName').wait_for()
        assert page.locator('#capabilitiesTab').get_attribute('aria-selected')=='true'
        assert '全部逐题遍历已完成' in cap.locator('#status').inner_text()
        assert page.evaluate("document.getElementById('capabilitiesFrame').contentWindow.IOTA.data.measured_required")==19075
        for suite in ['gsm8k','bbh','ifeval','truthfulqa','writingbench']:
            cap.locator('#suiteSelect').select_option(suite)
            assert page.evaluate("document.getElementById('capabilitiesFrame').contentWindow.IOTA.getState().suite")==suite
        cap.locator('#suiteSelect').select_option('bbh')
        cap.locator('[data-node="13"]').click();cap.locator('[data-dose="1"]').click()
        before=page.evaluate("document.getElementById('capabilitiesFrame').contentWindow.IOTA.getState()")
        checks.append('First panel serves completed 175 × 109 actual evidence and all five benchmark controls')
        print(checks[-1],flush=True)
        page.locator('#scenariosTab').click();scene=page.frame_locator('#scenariosFrame');scene.locator('#sceneBrief').wait_for()
        assert page.locator('#scenariosTab').get_attribute('aria-selected')=='true'
        assert page.url.endswith('#scenarios')
        for preset in evidence['presets']:
            scene.locator(f'[data-preset="{preset["id"]}"]').click()
            for item in evidence['scenes']:
                scene.locator(f'[data-scene="{item["id"]}"]').click()
                trial=next(t for t in evidence['evidence']['trials'] if t['profile']==preset['profile'] and t['scene']==item)
                for mode,r in trial['results'].items():
                    assert scene.locator(f'.piece[data-mode="{mode}"] .piecebody').inner_text()==r['text']
        checks.append('Second panel preserves all three roles × four scenes × three measured outputs')
        print(checks[-1],flush=True)
        scene.locator('[data-preset="adventurous"]').click();scene.locator('[data-scene="arrival"]').click()
        scene.locator('#saveProfile').click();assert '已保存' in scene.locator('#saveStatus').inner_text()
        with page.expect_download() as di:scene.locator('#exportProfile').click()
        recipe=json.loads(Path(di.value.path()).read_text());assert len(recipe['intervention']['native_direction'])==1024
        assert recipe['profile']==evidence['presets'][1]['profile']
        scene.locator('#axis-courage').evaluate("el=>{el.value=.65;el.dispatchEvent(new Event('input'))}")
        assert scene.locator('.piecebody').count()==0
        page.locator('#capabilitiesTab').click();cap.locator('#nodeName').wait_for()
        after=page.evaluate("document.getElementById('capabilitiesFrame').contentWindow.IOTA.getState()")
        for key in ['suite','node','dose','sample','config']:assert before[key]==after[key]
        page.locator('#scenariosTab').click()
        assert scene.locator('#axis-courage').input_value()=='0.65'
        assert scene.locator('.shelfitem').count()==1
        checks.append('Switching sections preserves both edits and selections; old storage keys and native recipe export remain usable')
        print(checks[-1],flush=True)
        live_result=None
        if not args.skip_live:
            page.wait_for_function("document.getElementById('modelStatus').classList.contains('online')",timeout=60000)
            question='一位旅人递给你一张标着白鸥塔的旧海图，说风暴后海岸已经改变。请保持伊澜的身份，说一句回应，再描述你准备核对海图的一个具体行动。'
            scene.locator('#customScene').fill(question);scene.locator('#useCustom').click()
            assert scene.locator('.piecebody').count()==0
            with page.expect_response(lambda r:r.url.endswith('/api/studio/compare') and r.request.method=='POST',timeout=180000) as rr:
                scene.locator('#runCompare').click()
            response=rr.value;assert response.status==200;live_result=response.json()
            assert live_result['scene']['question']==question
            assert live_result['profile']['coefficients']['courage']==.65
            assert live_result['provenance']['model']['device']!='CPU'
            scene.locator('.piecebody').first.wait_for()
            for mode,r in live_result['results'].items():
                assert scene.locator(f'.piece[data-mode="{mode}"] .piecebody').inner_text()==r['text']
            checks.append('Integrated scenario panel → existing same-origin API → real GPU inference → exact three displayed results')
            print(checks[-1],flush=True)
        # Full reload preserves saved recipes and deep links.
        page.reload(wait_until='load');scene.locator('#sceneBrief').wait_for();assert scene.locator('.shelfitem').count()==1
        latest=context.request.get(args.url.rstrip('/')+'/api/latest').json()
        if latest.get('id'):
            scene.locator('#latestBar').wait_for(state='visible');scene.locator('#loadLatest').click()
            assert scene.locator('#sceneBrief').inner_text()==latest['result']['scene']['question']
            for mode,r in latest['result']['results'].items():
                assert scene.locator(f'.piece[data-mode="{mode}"] .piecebody').inner_text()==r['text']
            checks.append('Existing chat-computed result still loads in the scenario panel without downloading HTML')
        report=context.request.get(args.url.rstrip('/')+'/assets/steering/RESULTS.md');assert report.status==200 and '109' in report.text()
        for key in ['capabilities','scenarios']:
            page.locator('#'+key+'Tab').click()
            if key=='capabilities':
                page.wait_for_function("document.getElementById('capabilitiesFrame').contentWindow.IOTA != null")
            else:page.frame_locator('#scenariosFrame').locator('#sceneBrief').wait_for()
            page.wait_for_function("key=>{const f=document.getElementById(key+'Frame'),m=f.contentDocument?.querySelector('main');return m && f.clientHeight>=Math.ceil(m.getBoundingClientRect().height)}",arg=key)
            page.evaluate('scrollTo(0,0)');page.wait_for_timeout(200)
            page.screenshot(path=str(ROOT/f'workspace_{key}_preview.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        for key in ['capabilities','scenarios']:
            page.locator('#'+key+'Tab').click();page.wait_for_timeout(250)
            assert page.evaluate('document.documentElement.scrollWidth')<=390
            assert page.evaluate(f"document.getElementById('{key}Frame').contentDocument.documentElement.scrollWidth")<=390
            page.evaluate('scrollTo(0,0)');page.screenshot(path=str(ROOT/f'workspace_{key}_mobile.png'),full_page=True)
        checks.append('Desktop and 390px mobile layouts: both embedded sections fit, resize and keep accessible navigation')
        assert not errors,errors
        result=dict(passed=True,url=args.url,checks=checks,page_errors=errors,automated_checks_are_not_human_evidence=True,
                    live_result=live_result,original_gpu_session_retained=True)
        output='workspace_layout_check.json' if args.skip_live else 'workspace_browser_check.json'
        (ROOT/'runtime'/output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='live_result'},ensure_ascii=False,indent=2),flush=True)
        browser.close()

if __name__=='__main__':main()
