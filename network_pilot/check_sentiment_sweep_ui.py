"""Exercise the measured sweep, split controls, linked charts, matrix and mobile."""
import json
import threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent;LAB=ROOT.parent/'steering_lab'
OUT=ROOT/'runs/sentiment9b-sst5-direction-sweep-v1'


def main():
    s=json.loads((OUT/'sweep-summary.json').read_text())
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(LAB/'dist'),**kw)
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport=dict(width=1440,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
            page.route('**/api/**',lambda route:route.fulfill(json=dict(online=True,latest=None)))
            page.goto(f'http://127.0.0.1:{server.server_port}/#capabilities')
            f=page.frame_locator('#capabilitiesFrame');f.locator('#sweepMatrix tbody tr[data-id]').first.wait_for()
            diagnostic_path=OUT.parent/'sentiment9b-sst5-diagnostics-v1/summary.json'
            if diagnostic_path.exists():
                diagnostic=json.loads(diagnostic_path.read_text())
                assert f.locator('#sentimentDiagnostics').is_visible()
                assert f.locator('#diagnosticConditions tbody tr').count()==10
                assert '100/100' in f.locator('#diagnosticFormat').inner_text()
                assert str(diagnostic['adjacent_class_errors']) in f.locator('#diagnosticGranularity').inner_text()
                assert '67/100' in f.locator('#diagnosticCoverage').inner_text()
                assert '89/100' in f.locator('#diagnosticWarmth').inner_text()
            for split in ['all','train','test']:
                f.locator('#sweepSplit').select_option(split);v=s['splits'][split]
                assert f.locator('#sweepBase').inner_text()==f"{v['baseline']['accuracy']:.1%}"
                assert f.locator('#sweepBest').inner_text()==f"{v['best_fixed_single']['accuracy']:.1%}"
                assert f.locator('#sweepOracle').inner_text()==f"{v['coverage']['single_directions']['accuracy']:.1%}"
                assert f.locator('#sweepMatrix tbody tr[data-id]').count()==v['n']
                assert f.locator('#sweepDirections tbody tr').count()==9
                assert f.locator('#sweepDoseChart polyline').count()==9
                assert f.locator('#sweepHistogram rect').count()==37
            f.locator('#sweepFilter').select_option('rescued')
            assert f.locator('#sweepMatrix tbody tr[data-id]').count()==s['splits']['test']['coverage']['single_directions']['rescued_baseline_errors']
            f.locator('#sweepFilter').select_option('all');f.locator('#sweepSplit').select_option('all')
            f.locator('#sweepAxis').select_option('warmth')
            assert f.locator('#sweepDoseChart polyline').count()==1
            assert f.locator('#sweepEffects tbody tr').count()==4
            f.locator('#sweepAxis').select_option('all')
            f.locator('#sweepGridMode').select_option('all')
            assert f.locator('#sweepMatrix tbody tr[data-id]').first.locator('td').count()==51
            f.locator('#sweepGridMode').select_option('single')
            assert f.locator('#sweepMatrix tbody tr[data-id]').first.locator('td').count()==39
            f.locator('#sweepMatrix tbody tr[data-id] button').first.click()
            assert f.locator('#sweepItemDetail').is_visible()
            assert '可行单方向' in f.locator('#sweepItemDetail').inner_text()
            f.locator('[aria-label="检查样本的配置概率"]').select_option('4')
            assert 'A–E 概率' in f.locator('#sweepItemDetail').inner_text()
            first=f.locator('#sweepMatrix tbody tr[data-id]').first.get_attribute('data-id')
            f.locator('#sweepQuery').fill(first)
            assert f.locator('#sweepMatrix tbody tr[data-id]').count()==1
            f.locator('#sweepQuery').fill('')
            f.locator('#sweepHistogram rect').first.click(force=True)
            expected=s['splits']['all']['coverage']['single_directions']['histogram'][0]['samples']
            assert f.locator('#sweepMatrix tbody tr[data-id]').count()==expected
            f.locator('#sweepFilter').select_option('all')
            f.locator('#priorExperiment summary').first.click()
            assert f.locator('#scores tbody tr').count()==8
            assert f.locator('#items tbody tr').count()==30
            f.locator('#priorExperiment summary').first.click()
            page.evaluate('scrollTo(0,0)')
            settled="""() => {const f=document.getElementById('capabilitiesFrame');const m=f.contentDocument.querySelector('main');return Math.abs(f.getBoundingClientRect().height-Math.ceil(m.getBoundingClientRect().bottom+f.contentWindow.scrollY+24))<2}"""
            page.wait_for_function(settled)
            page.screenshot(path=str(OUT/'navigator-desktop.png'),full_page=True)
            page.locator('#scenariosTab').click();search=page.frame_locator('#scenariosFrame');search.locator('#chatInput').wait_for()
            search.locator('#agentInnerTab').click();search.locator('#roleName').fill('方向扫描验证')
            page.locator('#capabilitiesTab').click();page.locator('#scenariosTab').click()
            assert search.locator('#roleName').input_value()=='方向扫描验证'
            page.locator('#capabilitiesTab').click();page.set_viewport_size(dict(width=390,height=844));page.evaluate('scrollTo(0,0)')
            assert page.evaluate('document.documentElement.scrollWidth')<=390
            assert page.evaluate("document.getElementById('capabilitiesFrame').contentDocument.documentElement.scrollWidth")<=390
            page.wait_for_function(settled)
            page.screenshot(path=str(OUT/'navigator-mobile.png'),full_page=True)
            page.goto(f'http://127.0.0.1:{server.server_port}/router-analysis.html')
            page.locator('#sweepBase').wait_for();assert page.locator('#sweepBase').inner_text()==f"{s['splits']['all']['baseline']['accuracy']:.1%}"
            assert not errors,errors
            result=dict(passed=True,page_errors=errors,checks=['all/train/test three requested metrics match measurements',
                '9 dose-response curves and signed comparison','36- and 48-configuration matrix','query and rescue filters',
                'linked histogram selection','per-item feasible directions and probabilities','prior fixed-policy results preserved',
                'Search settings preserved across tabs','390px mobile without overflow','Router alias shares sweep view'])
            (OUT/'ui-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
            browser.close()
    finally:server.shutdown();server.server_close()


if __name__=='__main__':main()
