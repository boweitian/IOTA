"""Validate the measured report and exact four-field sample rendering."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/classification9b-anli-r3-analysis-100-v1'

def main():
    data=json.loads((RUN/'view-data.json').read_text());summary=json.loads((RUN/'generation-summary.json').read_text())
    access=json.loads(Path('/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab/runtime/public-demo/access.json').read_text());url='http://127.0.0.1:8772'
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox']);ctx=browser.new_context(http_credentials=access,viewport={'width':1440,'height':1100});page=ctx.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url+'/#capabilities',wait_until='networkidle');nav=page.frame_locator('#capabilitiesFrame');nav.locator('#scores .big').first.wait_for();assert nav.locator('#doseRows tr').count()==10
        for split,n in [('all',100),('train',70),('test',30)]:
            nav.locator('#split').select_option(split);expected=summary['results']['0']['splits'][split]['accuracy'];assert nav.locator('#scores .big').first.inner_text()==f'{expected:.1%}';assert nav.locator('#matrixBody tr').count()==n
        nav.locator('#split').select_option('all');assert nav.locator('#matrixHead th').count()==55;page.screenshot(path=str(ROOT/'generation-desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':1000});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(ROOT/'generation-mobile.png'),full_page=True)
        page.goto(url+'/assets/navigator9b/anli-r3-analysis/samples.html?sample=21&condition=52',wait_until='networkidle');page.locator('#output').wait_for()
        assert page.locator('section h2').all_text_contents()==['1. 题目','2. 标准答案','3. 预测 prompt','4. 模型答案'];assert page.locator('#condition option').count()==54;assert page.locator('#sample option').count()==100
        row=data['rows'][20]
        for j in range(54):
            page.select_option('#condition',str(j));assert page.locator('#output').text_content()==row['outputs'][j]['text']
            prompt=data['actions'][j]['prompt_template'].replace('{context}',json.dumps(row['context'],ensure_ascii=False)).replace('{hypothesis}',json.dumps(row['hypothesis'],ensure_ascii=False));assert page.locator('#prompt').text_content()==prompt
        page.select_option('#condition','0');page.check('#wrongOnly');assert page.locator('#sample option').count()==37;page.uncheck('#wrongOnly');page.select_option('#sample','23');assert page.locator('#reasonBox').is_hidden()
        page.select_option('#sample','20');page.select_option('#condition','52');assert page.locator('#prediction').inner_text()=='A · 蕴含';page.evaluate('scrollTo(0,0)');assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(ROOT/'sample-analysis-mobile.png'),full_page=True)
        assert not errors;result={'passed':True,'checks':['all/train/test metrics match generated answers','54-configuration matrix and 10 signed curves','same four-field sample format','54 exact prompts and raw outputs checked for petition item','37 baseline wrong-item filter','missing official reason omitted','390px mobile layout'],'page_errors':errors}
        (ROOT/'generation-ui-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2));browser.close()

if __name__=='__main__':main()
