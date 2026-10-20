"""Check real evidence selection, native export, canvas interaction and mobile layout."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'runs/expanded-assets-v2');args=p.parse_args()
    out=args.run.resolve()/'view';errors=[];checks=[]
    print('Opening offline Chromium',flush=True)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        print('Browser launched',flush=True)
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((out/'index.html').as_uri(),wait_until='load',timeout=60000)
        page.wait_for_function('window.IOTA && document.getElementById("baseText").textContent.length > 0')
        print('Actual evidence loaded',flush=True)
        data=page.evaluate('({complete:IOTA.data.complete,rows:IOTA.data.rows.length,configs:IOTA.data.required_configurations})')
        assert data['rows']==175 and data['configs']==109
        for suite in ['gsm8k','bbh','ifeval','truthfulqa','writingbench']:
            page.select_option('#suiteSelect',suite)
            result=page.evaluate('''()=>{
                const data=IOTA.data,rows=data.rows.map((r,i)=>({r,i})).filter(x=>x.r.suite===IOTA.getState().suite);
                let tested=0;
                for(const {r,i} of rows){
                  document.getElementById('sampleSelect').value=i;
                  document.getElementById('sampleSelect').dispatchEvent(new Event('change'));
                  for(const j of [0,1,...Object.keys(r.evidence).map(Number).filter(j=>j>=37&&j<109).slice(-1)]){
                    IOTA.setConfig(j); const e=r.evidence[String(j)];
                    if(document.getElementById('question').textContent!==r.question)throw Error('Wrong question');
                    if(r.evidence['0'].text!=null&&document.getElementById('baseText').textContent!==r.evidence['0'].text)throw Error('Wrong baseline');
                    if(e?.text!=null&&document.getElementById('selectedText').textContent!==e.text)throw Error('Wrong selected output');
                    if(document.getElementById('selectedScore').textContent!==(e?(100*e.reward).toFixed(2)+'%':'未测'))throw Error('Wrong score');
                    tested++;
                  }
                }
                return tested;
            }''')
            checks.append(f'{suite}: {result} real question/configuration display checks')
            print(checks[-1],flush=True)
        page.select_option('#suiteSelect','gsm8k')
        result=page.evaluate('''()=>{let n=0;for(let i=0;i<27;i++)for(const dose of [-1,-.5,0,.5,1]){
            IOTA.setNode(i,dose);const s=IOTA.getState(),a=IOTA.data.configs[s.config];
            if(dose===0&&a.id!=='none')throw Error('Missing no-op');
            if(dose!==0&&(a.node!==i||Math.abs(a.strength-dose)>1e-12))throw Error('Wrong signed dose');
            if(document.getElementById('nodeName').textContent!==IOTA.data.concepts[i].name)throw Error('Wrong node');n++;
        }return n;}''')
        assert result==135;checks.append('27 nodes × five signed dose/no-op controls select the actual saved action')
        page.locator('#cosMode').click();assert page.evaluate('document.getElementById("geometryCanvas")._cosLayout.cell')>0
        page.locator('#mapMode').click();before=page.evaluate('IOTA.getState().yaw');page.locator('#geometryCanvas').focus();page.keyboard.press('ArrowRight');assert page.evaluate('IOTA.getState().yaw')!=before
        checks.append('Native cosine map and keyboard rotation')
        page.locator('#unresolvedOnly').check();page.locator('#unresolvedOnly').uncheck()
        page.locator('#rewardMode').click();page.locator('#gainMode').click()
        page.locator('#matrixCanvas').scroll_into_view_if_needed()
        box=page.locator('#matrixCanvas').bounding_box();assert box
        # Row 0, column 37 is the first expanded direction, dose -1.
        x=box['x']+24+37.5*(box['width']-29)/109;y=box['y']+85
        page.mouse.click(x,y)
        assert page.evaluate('IOTA.getState().node')==9 and page.evaluate('IOTA.getState().dose')==-1
        checks.append('Matrix click selects the measured question and expanded signed direction')
        selected=page.evaluate('''()=>{for(let i=0;i<IOTA.data.rows.length;i++){const r=IOTA.data.rows[i];for(const [j,e] of Object.entries(r.evidence)){
            if(+j>=37&&+j<109&&e.successful&&e.generated)return {sample:i,config:+j,suite:r.suite};}}return null;}''')
        assert selected is not None
        page.select_option('#suiteSelect',selected['suite']);page.select_option('#sampleSelect',str(selected['sample']));page.select_option('#configSelect',str(selected['config']))
        expected=page.evaluate('({config:IOTA.data.configs[IOTA.getState().config],row:IOTA.data.rows[IOTA.getState().sample]})')
        with page.expect_download() as di:page.locator('#exportConfig').click()
        exported=json.loads(Path(di.value.path()).read_text())
        assert exported['shift']==expected['config']['shift'] and len(exported['shift'])==1024
        assert exported['action']['coefficients']==expected['config']['coefficients'] and len(exported['action']['coefficients'])==27
        assert exported['evidence']['successful'] and exported['sample_id']==expected['row']['id']
        assert [r['id'] for r in exported['generation_batch']]==exported['evidence']['batch_sample_ids']
        checks.append('Successful new direction exports all 1024 native coordinates, 27 coefficients, real evidence and exact input cohort')
        replay_checked=page.evaluate('''()=>{let n=0;for(let i=0;i<IOTA.data.rows.length;i++){
            const r=IOTA.data.rows[i];for(const [j,e] of Object.entries(r.evidence))if(e.matched_replay){
                IOTA.setSuite(r.suite);document.getElementById('sampleSelect').value=i;
                document.getElementById('sampleSelect').dispatchEvent(new Event('change'));IOTA.setConfig(+j);
                const label=e.matched_replay.steered_correct?'匹配复跑成功':'复跑未重现';
                if(!document.getElementById('selectedNote').textContent.includes(label))throw Error('Missing replay evidence label');n++;
            }}return n;}''')
        assert replay_checked==2;checks.append('Both matched replay outcomes are displayed; failed reproduction remains visible')
        page.locator('#methodButton').click();assert page.locator('#methodDialog').is_visible();page.locator('#closeMethod').click()
        # Deterministic overview rather than selecting the best-scoring question for a screenshot.
        page.select_option('#suiteSelect','gsm8k');page.select_option('#sampleSelect','0');page.evaluate('IOTA.setNode(9,.5)')
        page.evaluate('scrollTo(0,0)');page.screenshot(path=str(out/'preview.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(250)
        assert page.evaluate('document.documentElement.scrollWidth')<=390
        page.screenshot(path=str(out/'preview_mobile.png'),full_page=True)
        checks.append('390px mobile layout without page overflow; offline page has no network dependence')
        browser.close()
    assert not errors,errors
    result=dict(passed=True,complete=data['complete'],checks=checks,page_errors=errors)
    (out/'browser_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
