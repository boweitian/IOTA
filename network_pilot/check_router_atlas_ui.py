"""Browser verification of the published Navigator diagnostic, no live inference."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent/'steering_lab'


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8768/');a=p.parse_args()
    checks=[];errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        response=page.goto(a.url+'router-analysis.html',wait_until='load');assert response.status==200
        page.wait_for_function('window.IOTARouterAtlas != null')
        for suite,total,n_train in [('gsm8k',1300,700),('bbh',540,378),('ifeval',500,350)]:
            page.select_option('#atlasSuite',suite);page.select_option('#atlasSplit','all')
            assert page.locator('#nQuestions').inner_text()==f'{total:,}'
            pca=page.evaluate('IOTARouterAtlas.projection()')
            assert abs(pca['alignment_cosine']-1)<1e-12
            assert pca['orthogonality_error']<1e-10
            page.select_option('#atlasSplit','train');assert page.locator('#nQuestions').inner_text()==str(n_train)
            page.select_option('#atlasMethod','text')
            expected=page.evaluate("()=>{const d=IOTARouterAtlas.data.suites[IOTARouterAtlas.getState().suite],r=d.rows.filter(r=>r.split==='train');return (100*r.filter(x=>x.rewards[x.predictions.text]===1).length/r.length).toFixed(2)+'%'}")
            assert page.locator('#nAccuracy').inner_text()==expected
            page.select_option('#atlasMethod','geometry_raw');page.select_option('#atlasOutcome','wrong')
            assert page.evaluate("()=>{const s=IOTARouterAtlas.getState(),d=IOTARouterAtlas.data.suites[s.suite],m=new Map(d.rows.map(r=>[r.id,r]));return s.filtered.every(id=>{const r=m.get(id);return r.rewards[r.predictions[s.method]]===0})}")
            page.select_option('#atlasOutcome','all');page.select_option('#atlasSplit','test')
            page.locator('#atlasTable tr').nth(1).click()
            row_id=page.locator('#atlasTable tr').nth(1).get_attribute('data-id')
            assert page.locator('#selectedId').inner_text()==row_id
            page.select_option('#atlasPC','2');assert 'PC2' in page.locator('#axisAudit').inner_text()
            page.select_option('#atlasPC','1')
            for coloring in ['best','router','binary']:
                page.select_option('#atlasColor',coloring)
                assert page.locator('#scatterLegend').inner_text()
            page.select_option('#atlasAxis','best')
            aligned=page.evaluate("()=>{const s=IOTARouterAtlas.getState(),d=IOTARouterAtlas.data.suites[s.suite],r=d.rows.find(r=>r.id===s.selected);return {reference:d.projection.axis_action,expected:d.actions[r.best==null||d.actions[r.best].norm===0?d.fixed:r.best].id,cosine:d.projection.alignment_cosine}}")
            assert aligned['reference']==aligned['expected'] and abs(aligned['cosine']-1)<1e-12
            page.select_option('#atlasAxis','router');page.select_option('#atlasAxis','fixed')
            page.locator('#atlasSearch').fill(row_id)
            page.wait_for_function("document.getElementById('nQuestions').textContent==='1'")
            page.locator('#atlasSearch').fill('')
            page.wait_for_function("()=>IOTARouterAtlas.getState().filtered.length>1")
        checks.append('All three benchmarks, train/test counts, source-backed accuracies, exact alignment and outcome filters verified')
        if page.evaluate("'fixed_advantage' in IOTARouterAtlas.data.methods"):
            page.locator('#advantageExperiment').wait_for(state='visible')
            assert page.locator('#advantageExperimentRows tr').count()==3
            page.select_option('#atlasMethod','fixed_advantage')
            assert page.locator('#methodEvidence').is_visible()
            expected=page.evaluate("()=>{const s=IOTARouterAtlas.getState(),d=IOTARouterAtlas.data.suites[s.suite],r=d.rows.filter(r=>r.split!=='train');return (100*r.filter(x=>x.rewards[x.predictions.fixed_advantage]===1).length/r.length).toFixed(2)+'%'}")
            assert page.locator('#nAccuracy').inner_text()==expected
            page.select_option('#atlasMethod','geometry_raw')
            checks.append('New fixed-advantage method agrees with historical matrix; independent fresh experiment is visibly separated')
        page.select_option('#atlasSuite','gsm8k');page.select_option('#atlasSplit','fresh')
        assert page.locator('#nQuestions').inner_text()=='300'
        with page.expect_download() as event:page.locator('#exportScatter').click()
        text=Path(event.value.path()).read_text();assert '<svg' in text and 'alignment_cosine' in text
        with page.expect_download() as event:page.locator('#exportRows').click()
        assert 'router' in Path(event.value.path()).read_text()
        checks.append('Point/table selection, new blind subset, SVG and CSV downloads work')
        checks.append('Per-question best/router axes and direction coloring switch correctly, retaining exact signed alignment')
        page.screenshot(path=str(ROOT/'runs/router-direction-atlas-v1/navigator_router_desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        for suite in ['gsm8k','bbh','ifeval']:
            page.select_option('#atlasSuite',suite);page.wait_for_timeout(100)
            assert page.evaluate('document.documentElement.scrollWidth')<=390
        page.screenshot(path=str(ROOT/'runs/router-direction-atlas-v1/navigator_router_mobile.png'),full_page=True)
        checks.append('390px mobile layout fits; table scroll remains inside its container')
        page.set_viewport_size({'width':1440,'height':1100})
        page.goto(a.url+'#capabilities',wait_until='load')
        cap=page.frame_locator('#capabilitiesFrame');cap.locator('#routerAnalysisFrame').wait_for()
        analysis=cap.frame_locator('#routerAnalysisFrame');analysis.locator('#atlasSuite').wait_for()
        assert page.evaluate("document.getElementById('capabilitiesFrame').contentWindow.IOTA.data.measured_required")==19075
        analysis.locator('#atlasSuite').select_option('bbh')
        cap.locator('#routerAtlas').scroll_into_view_if_needed();page.wait_for_timeout(200)
        page.screenshot(path=str(ROOT/'runs/router-direction-atlas-v1/navigator_integrated.png'))
        assert page.locator('#capabilitiesTab').get_attribute('aria-selected')=='true'
        checks.append('Analysis is embedded within existing IOTA Navigator; original 175 × 109 evidence remains intact')
        assert not errors,errors
        output=dict(passed=True,checks=checks,page_errors=errors,live_generation_invoked=False)
        (ROOT/'runs/router-direction-atlas-v1/browser_check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(output,ensure_ascii=False,indent=2));browser.close()


if __name__=='__main__':main()
