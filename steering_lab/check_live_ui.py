import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
errors=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
    page=browser.new_page(viewport={"width":1360,"height":960})
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto("http://127.0.0.1:8768")
    question="Write two short sentences from the perspective of a lighthouse. Use one vivid sound and one feeling."
    page.fill("#liveQuestion",question)
    with page.expect_response(lambda r:r.url.endswith("/api/route") and r.request.method=="POST",timeout=180000) as response:
        page.locator("#liveButton").click()
    r=response.value
    assert r.status==200
    actual=r.json()
    page.wait_for_function("document.getElementById('liveStatus').textContent.includes('实测完成')",timeout=10000)
    assert actual["question"]==question and actual["domain"]=="creative"
    assert actual["score"] is None and len(actual["features"])==18
    assert page.locator("#liveBase").inner_text()==actual["baseline"]["text"]
    assert page.locator("#liveSteered").inner_text()==actual["steered"]["text"]
    assert not page.locator("#liveButton").is_disabled()
    assert not errors,errors
    page.locator("#liveResults").screenshot(path=str(ROOT/"preview_live.png"))
    (ROOT/"live_browser_check.json").write_text(json.dumps(dict(actual_response=actual,page_errors=errors,
        checks=["real browser POST reaches local Qwen inference", "fresh task routed without gold answer",
                "displayed responses equal real API outputs", "controls recover after generation"]),ensure_ascii=False,indent=2))
    print("LIVE BROWSER OK",actual["action"],actual["baseline"]["text"],actual["steered"]["text"])
    browser.close()
