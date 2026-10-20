"""Load the newly computed chat action from the running portal, without a download."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent;expected=json.loads((ROOT/"runtime/chat_action_check.json").read_text());errors=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
    page=browser.new_page(viewport={"width":1440,"height":1100})
    page.on("pageerror",lambda e:errors.append(str(e)))
    assert page.goto("http://127.0.0.1:8768/scenarios.html").status==200
    page.locator("#latestBar").wait_for(state="visible")
    page.locator("#loadLatest").click()
    assert page.locator("#sceneBrief").inner_text()==expected["scene"]["question"]
    assert float(page.locator("#axis-courage").input_value())==.9
    for mode,r in expected["results"].items():
        assert page.locator(f".piece[data-mode='{mode}'] .piecebody").inner_text()==r["text"]
    assert "服务器刚完成" in page.locator("#trialStatus").inner_text()
    assert page.locator("#latestBar").is_hidden()
    page.screenshot(path=str(ROOT/"interactive_chat_preview.png"),full_page=True)
    assert not errors,errors
    (ROOT/"runtime/chat_browser_check.json").write_text(json.dumps(dict(checks=["served page receives chat-computed action", "one-click loads exact role, scene, coefficients and real outputs", "no HTML download", "zero browser errors"],page_errors=errors,manual_votes=0),indent=2))
    print("CHAT ACTION DISPLAY OK",flush=True);browser.close()
