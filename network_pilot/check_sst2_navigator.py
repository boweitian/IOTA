"""Actual binary results, signed plots, per-item evidence, archive and mobile UI."""
import json
import threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright
from sst2_prepare import OUT

ROOT=Path(__file__).resolve().parent;LAB=ROOT.parent/'steering_lab'

def main():
    summary=json.loads((OUT/'sweep-summary.json').read_text())
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(LAB/'dist'),**kw)
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport=dict(width=1440,height=1100))
            page.on('pageerror',lambda e:errors.append(str(e)));page.route('**/api/**',lambda r:r.fulfill(json=dict(online=True,latest=None)))
            page.goto(f'http://127.0.0.1:{server.server_port}/#capabilities');f=page.frame_locator('#capabilitiesFrame');f.locator('#sweepMatrix tbody tr[data-id]').first.wait_for()
            assert 'GLUE / SST-2' in f.locator('h1').inner_text()
            assert 'validation' in f.locator('main').inner_text()
            assert '100/100' in f.locator('#sst2Format').inner_text()
            assert page.evaluate('IOTAWorkspace.data.benchmark')=='SST-2'
            assert page.evaluate('IOTAWorkspace.data.shared_original_directions_bit_exact') is True
            for split in ['all','train','test']:
                f.locator('#sweepSplit').select_option(split);s=summary['splits'][split]
                for name,value in [('sweepBase',s['baseline']),('sweepBest',s['best_fixed_single']),('sweepOracle',s['coverage']['single_directions']),('sweepTrainBest',s['train_selected_single'])]:
                    assert f.locator('#'+name).inner_text()==f"{value['accuracy']:.1%}"
                assert f.locator('#sweepMatrix tbody tr[data-id]').count()==s['n']
                f.locator('#sweepFilter').select_option('rescued');assert f.locator('#sweepMatrix tbody tr[data-id]').count()==s['coverage']['single_directions']['rescued_baseline_errors']
                f.locator('#sweepFilter').select_option('all')
            f.locator('#sweepSplit').select_option('all')
            assert f.locator('#sweepDirections tbody tr').count()==9 and f.locator('#sweepDoseChart polyline').count()==9
            f.locator('#sweepAxis').select_option('warmth');assert f.locator('#sweepDoseChart polyline').count()==1 and f.locator('#sweepEffects tbody tr').count()==4
            f.locator('#sweepAxis').select_option('all');f.locator('#sweepGridMode').select_option('all')
            assert f.locator('#sweepMatrix tbody tr[data-id]').first.locator('td').count()==51
            f.locator('#sweepGridMode').select_option('single');f.locator('#sweepMatrix tbody tr button').first.click()
            assert 'A–B 概率' in f.locator('#sweepItemDetail').inner_text()
            select=f.locator('[aria-label="检查样本的配置概率"]');select.select_option('36')
            assert 'undefined' not in f.locator('#sweepItemDetail').inner_text()
            first=f.locator('#sweepMatrix tbody tr[data-id]').first.get_attribute('data-id');f.locator('#sweepQuery').fill(first)
            assert f.locator('#sweepMatrix tbody tr[data-id]').count()==1;f.locator('#sweepQuery').fill('')
            histogram=summary['splits']['all']['coverage']['single_directions']['histogram'];index=next(i for i,b in enumerate(histogram) if b['samples'])
            f.locator('#sweepHistogram rect').nth(index).click(force=True)
            assert f.locator('#sweepMatrix tbody tr[data-id]').count()==histogram[index]['samples']
            f.locator('#sweepFilter').select_option('all')
            f.locator('#sst2Geometry summary').click();assert f.locator('#gram button').count()==81
            f.locator('#gram button').nth(1).click();assert '余弦' in f.locator('#geometryNote').inner_text();f.locator('#sst2Geometry summary').click()
            settled="""() => {const f=document.getElementById('capabilitiesFrame');return Math.abs(f.getBoundingClientRect().height-Math.ceil(f.contentDocument.querySelector('main').getBoundingClientRect().bottom+f.contentWindow.scrollY+24))<2}"""
            page.wait_for_function(settled);page.evaluate('scrollTo(0,0)');page.screenshot(path=str(OUT/'navigator-desktop.png'),full_page=True)
            page.locator('#scenariosTab').click();search=page.frame_locator('#scenariosFrame');search.locator('#chatInput').wait_for()
            search.locator('#agentInnerTab').click();search.locator('#roleName').fill('SST2 界面验证');page.locator('#capabilitiesTab').click();page.locator('#scenariosTab').click()
            assert search.locator('#roleName').input_value()=='SST2 界面验证'
            page.locator('#capabilitiesTab').click();page.set_viewport_size(dict(width=390,height=844));page.evaluate('scrollTo(0,0)');page.wait_for_function(settled)
            assert page.evaluate('document.documentElement.scrollWidth')<=390
            assert page.evaluate("document.getElementById('capabilitiesFrame').contentDocument.documentElement.scrollWidth")<=390
            page.screenshot(path=str(OUT/'navigator-mobile.png'),full_page=True)
            page.goto(f'http://127.0.0.1:{server.server_port}/sst5-archive.html');page.locator('#sweepBase').wait_for();assert page.locator('#sweepBase').inner_text()=='48.0%'
            page.goto(f'http://127.0.0.1:{server.server_port}/router-analysis.html');page.locator('#sweepBase').wait_for();assert page.locator('#sweepBase').inner_text()==f"{summary['splits']['all']['baseline']['accuracy']:.1%}"
            assert not errors,errors;browser.close()
            result=dict(passed=True,page_errors=errors,checks=['official SST-2 and validation provenance displayed','all/train/evaluation requested scores match actual measurements',
                'binary per-item labels and A/B probabilities','49-configuration matrix and filters','9 signed dose curves and linked histogram',
                'same 9B geometry Gram inspection','Search state preserved','390px mobile layout','SST-5 archived with original scores','Router entry points to SST-2'])
            (OUT/'ui-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
