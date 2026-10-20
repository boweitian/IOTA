"""Calibrated NLI scores and controls, archived baselines, Search and mobile."""
import json
import threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
from anli_prepare import OUT

ROOT=Path(__file__).resolve().parent;LAB=ROOT.parent/'steering_lab'

def main():
    s=json.loads((OUT/'sweep-summary.json').read_text())
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(LAB/'dist'),**kw)
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport=dict(width=1440,height=1100))
            page.on('pageerror',lambda e:errors.append(str(e)));page.route('**/api/**',lambda r:r.fulfill(json=dict(online=True,latest=None)))
            page.goto(f'http://127.0.0.1:{server.server_port}/#capabilities');f=page.frame_locator('#capabilitiesFrame');f.locator('#sweepMatrix tbody tr[data-id]').first.wait_for()
            assert 'ANLI R3' in f.locator('h1').inner_text() and '17/30' in f.locator('#calibrationBadge').inner_text()
            assert '100/100' in f.locator('#classificationFormat').inner_text()
            assert page.evaluate('IOTAWorkspace.data.benchmark')=='ANLI R3'
            assert page.evaluate('IOTAWorkspace.data.shared_original_directions_bit_exact') is True
            for split in ['all','train','test']:
                f.locator('#sweepSplit').select_option(split);v=s['splits'][split]
                for element,value in [('sweepBase',v['baseline']),('sweepBest',v['best_fixed_single']),('sweepOracle',v['coverage']['single_directions']),('sweepTrainBest',v['train_selected_single'])]:
                    assert f.locator('#'+element).inner_text()==f"{value['accuracy']:.1%}"
                assert f.locator('#sweepMatrix tbody tr[data-id]').count()==v['n']
                f.locator('#sweepFilter').select_option('rescued');assert f.locator('#sweepMatrix tbody tr[data-id]').count()==v['coverage']['single_directions']['rescued_baseline_errors']
                f.locator('#sweepFilter').select_option('all')
            f.locator('#sweepSplit').select_option('all');f.locator('#sweepMatrix tbody tr button').first.click()
            assert 'A–C 概率' in f.locator('#sweepItemDetail').inner_text()
            assert 'Premise:' in f.locator('#sweepItemDetail').inner_text() and 'Hypothesis:' in f.locator('#sweepItemDetail').inner_text()
            assert 'undefined' not in f.locator('#sweepItemDetail').inner_text()
            f.locator('#sweepGridMode').select_option('all');assert f.locator('#sweepMatrix tbody tr[data-id]').first.locator('td').count()==51
            f.locator('#sweepGridMode').select_option('single');assert f.locator('#sweepDirections tbody tr').count()==9
            f.locator('#sweepAxis').select_option('confidence');assert f.locator('#sweepDoseChart polyline').count()==1
            f.locator('#sweepAxis').select_option('all');assert f.locator('#sweepDoseChart polyline').count()==9
            first=f.locator('#sweepMatrix tbody tr[data-id]').first.get_attribute('data-id');f.locator('#sweepQuery').fill(first);assert f.locator('#sweepMatrix tbody tr[data-id]').count()==1;f.locator('#sweepQuery').fill('')
            hist=s['splits']['all']['coverage']['single_directions']['histogram'];i=next(i for i,b in enumerate(hist) if b['samples'])
            f.locator('#sweepHistogram rect').nth(i).click(force=True);assert f.locator('#sweepMatrix tbody tr[data-id]').count()==hist[i]['samples'];f.locator('#sweepFilter').select_option('all')
            f.locator('#classificationGeometry summary').click();assert f.locator('#gram button').count()==81;f.locator('#gram button').nth(1).click();assert '余弦' in f.locator('#geometryNote').inner_text();f.locator('#classificationGeometry summary').click()
            settled="""() => {const f=document.getElementById('capabilitiesFrame');return Math.abs(f.getBoundingClientRect().height-Math.ceil(f.contentDocument.querySelector('main').getBoundingClientRect().bottom+f.contentWindow.scrollY+24))<2}"""
            page.wait_for_function(settled);page.evaluate('scrollTo(0,0)');page.screenshot(path=str(OUT/'navigator-desktop.png'),full_page=True)
            page.locator('#scenariosTab').click();search=page.frame_locator('#scenariosFrame');search.locator('#chatInput').wait_for();search.locator('#agentInnerTab').click();search.locator('#roleName').fill('ANLI 检查')
            page.locator('#capabilitiesTab').click();page.locator('#scenariosTab').click();assert search.locator('#roleName').input_value()=='ANLI 检查'
            page.locator('#capabilitiesTab').click();page.set_viewport_size(dict(width=390,height=844));page.evaluate('scrollTo(0,0)');page.wait_for_function(settled)
            assert page.evaluate('document.documentElement.scrollWidth')<=390 and page.evaluate("document.getElementById('capabilitiesFrame').contentDocument.documentElement.scrollWidth")<=390
            page.screenshot(path=str(OUT/'navigator-mobile.png'),full_page=True)
            for path,expected in [('sst2-archive.html','96.0%'),('sst5-archive.html','48.0%')]:
                page.goto(f'http://127.0.0.1:{server.server_port}/'+path);page.locator('#sweepBase').wait_for();assert page.locator('#sweepBase').inner_text()==expected
            page.goto(f'http://127.0.0.1:{server.server_port}/router-analysis.html');page.locator('#sweepBase').wait_for();assert 'ANLI R3' in page.locator('h1').inner_text()
            assert not errors,errors;browser.close()
            result=dict(passed=True,page_errors=errors,checks=['independent calibration score and task provenance','all/train/test requested metrics match measurements',
                'three-class per-item premise/hypothesis and probabilities','49 configurations and signed dose curves','filters and linked histogram','shared geometry inspection',
                'Search configuration preserved','390px mobile layout','SST-2 and SST-5 historical scores preserved','Router alias opens current task'])
            (OUT/'ui-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
