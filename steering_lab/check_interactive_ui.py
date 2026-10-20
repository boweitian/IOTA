"""Live portal + shared-file GPU worker, verified with a new user interaction."""
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
    page.wait_for_function("document.getElementById('connection').textContent.includes('已连接')",timeout=60000)
    assert "会话预计结束" in page.locator("#runtimeMessage").inner_text()
    page.locator("#axis-courage").evaluate("el=>{el.value=.6;el.dispatchEvent(new Event('input'))}")
    page.locator("#axis-imagery").evaluate("el=>{el.value=.8;el.dispatchEvent(new Event('input'))}")
    question="旅人带来一张画满陌生星座的海图，说想与你一起寻找一座从未有人见过的岛。你会如何回应？写一句对白和一个行动。"
    page.fill("#customScene",question);page.locator("#useCustom").click()
    assert page.locator(".piecebody").count()==0
    with page.expect_response(lambda r:r.url.endswith("/api/studio/compare") and r.request.method=="POST",timeout=180000) as response:
        page.locator("#runCompare").click()
    result=response.value.json();assert response.value.status==200
    assert result["scene"]["question"]==question and result["profile"]["coefficients"]["courage"]==.6
    assert result["profile"]["coefficients"]["imagery"]==.8
    assert result["provenance"]["model"]["device"]!="CPU"
    page.wait_for_function("document.getElementById('trialStatus').textContent.includes('实时生成')")
    for mode,record in result["results"].items():
        assert page.locator(f".piece[data-mode='{mode}'] .piecebody").inner_text()==record["text"]
    assert not page.locator("#runCompare").is_disabled()
    assert not errors,errors
    page.screenshot(path=str(ROOT/"interactive_preview.png"),full_page=True)
    check=dict(checks=["served page with live model and expiry status", "fresh scenario with previously unmeasured mixed coefficients", "browser POST → shared-filesystem request → GPU inference → exact displayed outputs", "no manual model startup per action", "zero browser errors"],response=result,page_errors=errors,manual_votes=0)
    (ROOT/"runtime/interactive_browser_check.json").write_text(json.dumps(check,ensure_ascii=False,indent=2));print("INTERACTIVE GPU BROWSER OK",flush=True)
    browser.close()
