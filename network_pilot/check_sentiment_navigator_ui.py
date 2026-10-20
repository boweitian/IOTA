"""Actual rendered results, per-item routing evidence, mobile layout and Search persistence."""
import json
import threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent;LAB=ROOT.parent/'steering_lab'

def main():
    summary=json.loads((ROOT/'runs/sentiment9b-sst5-100-v1/summary.json').read_text())
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(LAB/'dist'),**kw)
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport=dict(width=1440,height=1100))
            page.route('**/api/**',lambda route:route.fulfill(json=dict(online=True,latest=None)))
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/#capabilities')
            frame=page.frame_locator('#capabilitiesFrame');frame.locator('#scores tbody tr').first.wait_for()
            assert frame.locator('#scores tbody tr').count()==8
            assert frame.locator('#items tbody tr').count()==30
            expected=f"{100*summary['test_metrics'][summary['selected']]['accuracy']:.1f}%"
            assert frame.locator('#selectedScore').inner_text()==expected
            assert page.evaluate('IOTAWorkspace.data.shared_original_directions_bit_exact') is True
            assert page.evaluate('IOTAWorkspace.data.model')=='Qwen3.5-9B'
            assert summary['geometry_sha256'] in frame.locator('#binding').text_content()
            assert frame.locator('#gram button').count()==81
            frame.locator('#gram button').nth(1).click()
            assert '余弦' in frame.locator('#geometryNote').inner_text()
            for name,policy in summary['policies'].items():
                frame.locator('#policySelect').select_option(name)
                assert frame.locator('#members .member').count()==len(policy['members'])
                frame.locator('#correctFilter').select_option('wrong')
                assert frame.locator('#items tbody tr').count()==30-summary['test_metrics'][name]['correct']
                frame.locator('#correctFilter').select_option('all')
            frame.locator('#policySelect').select_option('router_3_10')
            frame.locator('#items tbody tr').first.click()
            assert '实际权重' in frame.locator('#itemDetail').inner_text()
            first=frame.locator('#items tbody tr td').first.inner_text()
            frame.locator('#query').fill(first)
            assert frame.locator('#items tbody tr').count()==1
            frame.locator('#query').fill('')
            page.screenshot(path=str(ROOT/'runs/sentiment9b-sst5-100-v1/navigator-desktop.png'),full_page=True)
            page.locator('#scenariosTab').click()
            search=page.frame_locator('#scenariosFrame');search.locator('#chatInput').wait_for()
            search.locator('#agentInnerTab').click();search.locator('#roleName').fill('界面验证')
            page.locator('#capabilitiesTab').click();page.locator('#scenariosTab').click()
            assert search.locator('#roleName').input_value()=='界面验证'
            page.locator('#capabilitiesTab').click()
            page.set_viewport_size(dict(width=390,height=844));page.evaluate('scrollTo(0,0)')
            assert page.evaluate('document.documentElement.scrollWidth')<=390
            assert page.evaluate("document.getElementById('capabilitiesFrame').contentDocument.documentElement.scrollWidth")<=390
            page.screenshot(path=str(ROOT/'runs/sentiment9b-sst5-100-v1/navigator-mobile.png'),full_page=True)
            page.goto(f'http://127.0.0.1:{server.server_port}/router-analysis.html')
            page.locator('#scores tbody tr').first.wait_for()
            assert 'Qwen3.5-9B' in page.locator('#modelBadge').inner_text()
            assert not errors,errors
            report=dict(passed=True,checks=['9B-only Navigator and Search share exact geometry','all eight fixed-policy scores match actual experiment',
                '30 holdout items, correct/error filters and search','Gram inspection and per-item dynamic member weights',
                'Search identity preserved across tab switches','390px mobile and desktop layout','old Router entry now opens the same 9B results'],page_errors=errors)
            (ROOT/'runs/sentiment9b-sst5-100-v1/ui-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(report,ensure_ascii=False,indent=2));browser.close()
    finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
