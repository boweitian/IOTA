"""Browser checks for the actual atlas, knob, export, and narrow-screen layout.

Run with Playwright available in the selected Python environment.
"""
import json
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent

def main():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1440,height=1050),device_scale_factor=1)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((ROOT/'dist/index.html').as_uri())
        page.wait_for_function("document.querySelectorAll('.axis-chip').length === 7")
        assert page.locator('#baseline-choice').input_value()=='-2'
        assert page.locator('#strength').input_value()=='2'
        assert '不会参与' in page.locator('#baseline').inner_text()
        assert '尝试' in page.locator('#steered').inner_text()
        page.locator('#reset').click()
        assert page.locator('#baseline').inner_text()==page.locator('#steered').inner_text()
        for direction in ['courage','warmth','curiosity','candor','confidence','patience','agency']:
            page.locator(f'[data-axis="{direction}"]').click()
            for strength in [-2,-1,0,1,2]:
                page.locator('#strength').fill(str(strength))
                assert page.locator('#dial').get_attribute('aria-valuenow')==str(strength)
                assert page.locator('#steered').inner_text()!='此方向与强度的回答尚未测量。'
                for scenario in ['challenge','disagreement','setback']:
                    page.locator(f'[data-scenario="{scenario}"]').click()
                    assert page.locator('#steered').inner_text()
            page.locator('#reset').click()
            assert page.locator('#baseline').inner_text()==page.locator('#steered').inner_text()
        page.locator('[data-axis="courage"]').click()
        page.locator('[data-scenario="challenge"]').click()
        page.locator('#dial').focus();page.keyboard.press('ArrowRight')
        assert page.locator('#dial').get_attribute('aria-valuenow')=='1'
        page.locator('#strength').fill('2')
        page.evaluate('state.busy=true')
        page.locator('#strength').fill('-2')
        assert page.locator('#strength').input_value()=='2'
        assert page.locator('#dial').get_attribute('aria-valuenow')=='2'
        page.evaluate('state.busy=false')
        page.locator('#measured').click()
        assert page.locator('#measured').get_attribute('aria-pressed')=='true'
        before=page.evaluate('state.yaw')
        box=page.locator('#globe').bounding_box()
        page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2)
        page.mouse.down();page.mouse.move(box['x']+box['width']/2+70,box['y']+box['height']/2+10,steps=5);page.mouse.up()
        assert page.evaluate('state.yaw')!=before
        page.screenshot(path=str(ROOT/'preview_measured.png'),full_page=True)
        with page.expect_download() as dl:
            page.locator('#export').click()
        with tempfile.TemporaryDirectory() as tmp:
            export=Path(tmp)/'direction.json';dl.value.save_as(export)
            config=json.loads(export.read_text())
            assert config['direction']=='courage' and config['strength']==2 and config['weights_modified'] is False
            assert len(config['delta_per_strength_unit'])==config['hidden_size']==896
        page.locator('#about').click();assert page.locator('#details').is_visible()
        page.keyboard.press('Escape');assert not page.locator('#details').is_visible()
        page.evaluate('state.yaw=25*RAD;state.pitch=10*RAD')
        page.locator('#intuitive').click();page.locator('#baseline-choice').select_option('-2');page.locator('#strength').fill('2')
        page.screenshot(path=str(ROOT/'preview.png'),full_page=True)
        # Hit testing uses rendered cloud centers, not arbitrary canvas coordinates.
        page.evaluate('redraw()');page.wait_for_timeout(100)
        hit=page.evaluate("hits.flat.find(p=>p.kind==='label'&&p.axis==='warmth')")
        rect=page.locator('#flat').bounding_box();page.mouse.click(rect['x']+hit['x'],rect['y']+hit['y'])
        assert page.locator('[data-axis="warmth"]').get_attribute('aria-pressed')=='true'
        page.set_viewport_size(dict(width=390,height=844))
        page.wait_for_timeout(100)
        assert page.evaluate('document.documentElement.scrollWidth')<=390
        page.locator('#strength').fill('-2')
        page.screenshot(path=str(ROOT/'preview_mobile.png'),full_page=True)
        assert not errors,errors
        print('PASS: 105 response settings; knob keyboard; map selection/rotation; projection switch; JSON export; modal; mobile width; no JS errors.')
        browser.close()

if __name__=='__main__':main()
