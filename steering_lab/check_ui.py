"""Browser checks against the actual completed evidence, never fabricated outputs."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument("--url",default=(ROOT/"dist/benchmark.html").as_uri());args=p.parse_args()
    evidence=json.loads((ROOT/"dist/evidence.json").read_text())
    errors=[];checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":1440,"height":1100},device_scale_factor=1)
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto(args.url)
        page.wait_for_function("document.getElementById('baseText').textContent.length > 0")
        for domain in ["creative","objective"]:
            page.locator(f"#domainTabs [data-domain='{domain}']").click()
            for i,e in enumerate(evidence[domain]["examples"]):
                page.select_option("#exampleSelect",str(i))
                for policy in ["geometry","fixed","lexical","prompt","random"]:
                    page.select_option("#policySelect",policy)
                    assert page.locator("#baseText").inner_text()==e["results"]["baseline"]["text"]
                    assert page.locator("#selectedText").inner_text()==e["results"][policy]["text"]
                    assert page.locator("#question").inner_text()==e["question"]
            checks.append(f"all {domain} held-out records × five policies")
        page.locator("#domainTabs [data-domain='creative']").click()
        for i,e in enumerate([e for e in evidence["dev_examples"] if e["kind"]=="creative"]):
            page.select_option("#devSelect",str(i))
            for c in evidence["atlas"]["concepts"]:
                page.select_option("#axisSelect",c["id"])
                for step,suffix in [(0,"none"),(1,c["id"]+":0.5"),(2,c["id"]+":1.0")]:
                    page.locator("#knob").evaluate("(el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}))}",step)
                    expected=next(v for v in e["variants"] if v["action"]==suffix)
                    assert page.locator("#knobOutput").inner_text()==expected["text"]
        checks.append("all creative development knobs map to actual saved generations")
        page.locator("#legend [data-concept='courage']").click()
        with page.expect_download() as di:
            page.locator("#exportDirection").click()
        download=di.value
        direction=json.loads(Path(download.path()).read_text())
        native=next(c for c in evidence["atlas"]["concepts"] if c["id"]=="courage")["native_direction"]
        assert direction["direction"]==native and len(native)==1024
        checks.append("export round-trips exact 1024-dimensional native direction")
        page.locator("#methodButton").click();assert page.locator("#methodDialog").is_visible()
        page.locator("#closeMethod").click()
        page.locator("#globe").focus();page.keyboard.press("ArrowRight")
        page.locator("#blindNext").click();page.locator("[data-vote='TIE']").click()
        assert "已记录" in page.locator("#blindStatus").inner_text()
        with page.expect_download() as di:
            page.locator("#exportVotes").click()
        votes=json.loads(Path(di.value.path()).read_text());assert votes[-1]["winner"]=="TIE"
        checks.append("rotation, methods dialog, local anonymous human vote export")
        page.locator("#blindNext").click()
        if evidence.get("budget_audit"):
            page.select_option("#budgetSelect","audit")
            for domain in ["creative","objective"]:
                page.locator(f"#domainTabs [data-domain='{domain}']").click()
                for i,e in enumerate(evidence["budget_audit"][domain]["examples"]):
                    page.select_option("#exampleSelect",str(i))
                    for policy in ["geometry","fixed","lexical","prompt","random"]:
                        page.select_option("#policySelect",policy)
                        assert page.locator("#selectedText").inner_text()==e["results"][policy]["text"]
                        assert page.locator("#baseText").inner_text()==e["results"]["baseline"]["text"]
            page.select_option("#budgetSelect","standard")
            page.locator("#domainTabs [data-domain='creative']").click()
            checks.append("all larger-budget audit examples × five policies; no stale output after switching budgets")
        # Display a predetermined item, not a score-selected showcase.
        page.select_option("#exampleSelect","0");page.select_option("#policySelect","geometry")
        page.select_option("#devSelect","0");page.select_option("#axisSelect","imagery")
        page.locator("#knob").evaluate("el=>{el.value=1;el.dispatchEvent(new Event('input',{bubbles:true}))}")
        page.screenshot(path=str(ROOT/"preview.png"),full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        page.wait_for_timeout(250)
        assert page.evaluate("document.documentElement.scrollWidth")<=390
        page.screenshot(path=str(ROOT/"preview_mobile.png"),full_page=True)
        checks.append("390px viewport without horizontal page overflow")
        browser.close()
    assert not errors,errors
    result=dict(checks=checks,page_errors=errors,protocol_sha256=evidence["protocol_sha256"])
    (ROOT/"browser_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
