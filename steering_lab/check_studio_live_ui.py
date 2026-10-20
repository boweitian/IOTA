"""Browser → local CPU service → real fresh character response."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
errors=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
    page=browser.new_page(viewport={"width":1440,"height":1100})
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto("http://127.0.0.1:8768/scenarios.html")
    page.wait_for_function("document.getElementById('connection').textContent.includes('已连接')")
    page.locator("details.editdetails").first.evaluate("el=>el.open=true")
    page.fill("#profileTitle","雨天书店 · 青禾")
    page.fill("#roleName","青禾")
    page.fill("#roleIdentity","你是书店店主青禾，喜欢观察雨天的城市。保持第一人称角色口吻。")
    page.fill("#roleFacts","书店叫纸舟。\n你的朋友叫小满。\n店里有一只叫墨点的猫。")
    page.fill("#roleBoundaries","保持角色事实，避免重复和无关解释。")
    page.fill("#customScene","墨点把书推到地上，客人笑了。你会说一句什么，再做一个动作？")
    page.locator("#useCustom").click()
    assert page.locator(".piecebody").count()==0
    with page.expect_response(lambda r:r.url.endswith("/api/studio/compare") and r.request.method=="POST",timeout=180000) as response:
        page.locator("#runCompare").click()
    actual=response.value.json();assert response.value.status==200
    page.wait_for_function("document.getElementById('trialStatus').textContent.includes('实时生成')")
    assert actual["profile"]["role"]["name"]=="青禾"
    for mode,record in actual["results"].items():
        assert page.locator(f".piece[data-mode='{mode}'] .piecebody").inner_text()==record["text"]
    assert page.locator("#runCompare").is_enabled()
    assert not errors,errors
    page.locator(".stage").screenshot(path=str(ROOT/"studio_preview_live.png"))
    result=dict(checks=["online status detects real local model", "custom role and scene clear archived output", "real browser POST generates three comparator outputs", "displayed texts equal raw API response", "controls recover after real inference"],response=actual,page_errors=errors,manual_votes=0)
    (ROOT/"studio_live_browser_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2));print("LIVE CHARACTER BROWSER OK",flush=True)
    browser.close()
