"""Actual browser flows, including saved recipes and prevention of stale evidence."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent

def main():
    data=json.loads((ROOT/"dist/studio-evidence.json").read_text());errors=[];checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1440,"height":1050},accept_downloads=True)
        page=context.new_page();page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto((ROOT/"dist/scenarios.html").as_uri())
        page.locator('#agentInnerTab').click();page.locator('.agent-tools').evaluate('el=>el.open=true')
        page.locator('#legacyDirectionControls').evaluate('el=>el.open=true')
        for preset in data["presets"]:
            page.locator(f"[data-preset='{preset['id']}']").click()
            for scene in data["scenes"]:
                page.locator(f"[data-scene='{scene['id']}']").click()
                actual=next(t for t in data["evidence"]["trials"] if t["profile"]==preset["profile"] and t["scene"]==scene)
                for mode,record in actual["results"].items():
                    assert page.locator(f".piece[data-mode='{mode}'] .piecebody").inner_text()==record["text"]
        checks.append("all three preset characters × four scenes × three actual methods")
        page.locator("[data-preset='adventurous']").click()
        page.locator("[data-scene='arrival']").click()
        page.locator("#axis-courage").evaluate("el=>{el.value=.15;el.dispatchEvent(new Event('input'))}")
        assert page.locator(".piecebody").count()==0 and "尚未实测" in page.locator("#compare").inner_text()
        checks.append("changing a direction clears old generation evidence")
        page.locator("#saveProfile").click();assert "已保存" in page.locator("#saveStatus").inner_text()
        with page.expect_download() as download: page.locator("#exportProfile").click()
        exported_bytes=Path(download.value.path()).read_bytes();exported=json.loads(exported_bytes)
        assert exported["profile"]["coefficients"]["courage"]==.15
        assert len(exported["intervention"]["native_direction"])==data["atlas"]["model"]["hidden_size"]
        intervention=exported["intervention"]
        if intervention.get('regularization')=='direction_projection':
            assert max(map(abs,intervention['projections'].values()),default=0.)<=intervention['projection_limit']+1e-5
        elif intervention.get('regularization','norm')=='norm':
            assert intervention['norm']<=intervention['cap']+1e-5
        else:
            assert intervention['cap_factor']==1.
        page.locator("#neutral").click()
        page.locator("#shelf button").filter(has_text="载入").click()
        assert page.locator("#axis-courage").input_value()=="0.15"
        page.reload();page.locator("#agentInnerTab").click();page.locator("#legacyDirectionControls").evaluate("el=>el.open=true");page.locator(".agent-tools").evaluate("el=>el.open=true");assert page.locator(".shelfitem").count()==1
        page.locator("#profileFile").set_input_files(dict(name="recipe.json",mimeType="application/json",buffer=exported_bytes))
        page.wait_for_function("document.getElementById('saveStatus').textContent.includes('已导入')")
        assert page.locator("#axis-courage").input_value()=="0.15"
        incompatible=dict(exported,binding=dict(exported["binding"],revision="wrong-version"))
        page.locator("#profileFile").set_input_files(dict(name="wrong.json",mimeType="application/json",buffer=json.dumps(incompatible).encode()))
        page.wait_for_function("document.getElementById('saveStatus').textContent.includes('不同')")
        assert page.locator("#axis-courage").input_value()=="0.15"
        checks.append("exact high-dimensional recipe export, local save/reload/import, model-version rejection")
        page.locator("[data-preset='adventurous']").click()
        page.locator("[data-scene='arrival']").click()
        page.locator("#toggleBlind").click();assert "作品 A" in page.locator("#compare").inner_text()
        assert page.locator(".ratings").count()==0
        page.locator("[data-vote='baseline']").click()
        with page.expect_download() as download: page.locator("#exportSession").click()
        session=json.loads(Path(download.value.path()).read_text())
        assert session["manual_reviews"][-1]["blind"] is False
        page.locator("[data-scene='choice']").click()
        page.locator("[data-vote='baseline']").click()
        with page.expect_download() as download: page.locator("#exportSession").click()
        session=json.loads(Path(download.value.path()).read_text())
        assert session["manual_reviews"][-1]["anonymous_display"] is True
        assert session["manual_reviews"][-1]["winner"] in ["baseline","prompt","geometry"]
        checks.append("anonymous output ordering, no revealed scores, previously revealed votes marked non-blind")
        page.locator("#adoptRecipe").click()
        selected=data["evidence"]["walker"]["winner"]["profile"]
        assert float(page.locator("#axis-courage").input_value())==selected["coefficients"]["courage"]
        assert page.locator(".piecebody").count()==3
        checks.append("adopt real frozen Walker recipe and replay its measured outputs")
        page.locator("#customScene").fill("一只海鸟落在窗台，你会如何回应？")
        page.locator("#useCustom").click()
        assert page.locator(".piecebody").count()==0 and "海鸟" in page.locator("#sceneBrief").inner_text()
        page.locator("#methodButton").click();assert page.locator("#methodDialog").is_visible();page.locator("#closeMethod").click()
        # A fresh context for screenshots: no QA clicks shown as human evidence.
        context.close();context=browser.new_context(viewport={"width":1440,"height":1050});page=context.new_page()
        page.on("pageerror",lambda e:errors.append(str(e)));page.goto((ROOT/"dist/scenarios.html").as_uri())
        page.screenshot(path=str(ROOT/"studio_preview.png"),full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth")<=390
        page.screenshot(path=str(ROOT/"studio_preview_mobile.png"),full_page=True)
        checks.append("method dialog, custom scene clearing, 390px layout without overflow")
        browser.close()
    assert not errors,errors
    result=dict(checks=checks,page_errors=errors,automated_checks_are_not_human_evidence=True)
    (ROOT/"studio_browser_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
