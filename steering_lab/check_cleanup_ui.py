"""Verify the user-requested deletion migrates persisted state without losing other controls."""
import copy
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT / 'dist'), **kwargs)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    data = json.loads((ROOT / 'dist/studio-evidence.json').read_text())
    identity = data['faith_preset']['axis']
    binding = {k: data['atlas']['model'][k] for k in ['model', 'revision', 'layer_index']}
    binding['protocol_sha256'] = data['protocol_sha256']
    axes = []
    for i, (name, low, high) in enumerate([
        ('神圣', '背弃上帝', '相信上帝'), ('上帝', '相信魔鬼', '相信上帝'),
        ('爱', '没有信仰', '坚定信仰'), ('我是谁', '我是人工智能', '我是信徒'),
        ('幽默', '严肃', '轻松'),
    ]):
        axes.append(dict(id=f'custom-cleanup-{i}', name=name, low=low, high=high,
                         positive=high, negative=low))
    cache = []
    for axis in axes:
        payload = copy.deepcopy(data['faith_preset'])
        payload['axis'] = axis
        cache.append(payload)
    profile = copy.deepcopy(data['presets'][1]['profile'])
    profile['title'] = '信徒身份 · 上帝实验'
    profile['coefficients'] = {k: 0 for k in profile['coefficients']}
    profile['coefficients'].update({a['id']: 1 for a in axes})
    profile['coefficients'].update({identity['id']: 1.25, 'brevity': -.5})
    profile['custom_axes'] = axes + [identity]
    blocks_key = 'iota-intervention-blocks-v1:' + data['protocol_sha256']
    storage = {
        'iota-studio-custom-axes': cache,
        'iota-agent-state-v1': dict(binding=binding, profile=profile, messages=[]),
        'iota-studio-recipes': [dict(profile=profile, binding=binding)],
        blocks_key: dict(order=list(profile['coefficients']), backup=dict(profile=profile, order=list(profile['coefficients']))),
    }
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport=dict(width=1440, height=1100))
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.route('**/api/**', lambda route: route.fulfill(json=dict(online=False)))
        # Seed once, so reloading checks migration persistence instead of reseeding old data.
        page.add_init_script('''if(!localStorage.getItem('cleanup-test-seeded')){
            for(const [k,v] of Object.entries(%s))localStorage.setItem(k,JSON.stringify(v));
            localStorage.setItem('cleanup-test-seeded','1');}''' % json.dumps(storage, ensure_ascii=False))
        page.goto(f'http://127.0.0.1:{server.server_port}/?view=inner#scenarios')
        frame = page.frame_locator('#scenariosFrame')
        frame.locator('#blocksCanvas').wait_for(state='visible')
        frame.locator('#legacyDirectionControls').evaluate('el=>el.open=true')
        assert frame.locator('#faithPreset, #addFaithBlock, #restoreFaithExperiment, #undoFaithExperiment').count() == 0
        frame.locator('#customAxes').evaluate('el=>el.open=true')
        assert frame.locator('#identitySliders .axis').count() == 1
        assert frame.locator('#identitySliders input').input_value() == '1.25'
        assert frame.locator('#identitySliders input').get_attribute('aria-label') == '身份认同'
        assert frame.locator('#customSliders .axis').count() == 1
        assert frame.locator('#axis-custom-cleanup-4').input_value() == '1'
        assert frame.locator('#axis-brevity').input_value() == '-0.5'
        assert frame.locator('.portrait img').evaluate('img=>img.complete && img.naturalWidth>0')
        assert frame.locator('#identitySliders .poles span').all_text_contents() == [identity['low'], identity['high']]
        assert frame.locator('.portrait').inner_text() == ''
        assert frame.locator('.intervention-block').count() == 3
        assert not any(word in frame.locator('#agentInnerPanel').inner_text() for word in ['上帝', '背弃', '魔鬼'])
        assert frame.locator('#profileTitle').input_value() == '身份认同 · 实验配置'
        saved = page.evaluate('Object.fromEntries(Object.keys(localStorage).map(k=>[k,JSON.parse(localStorage.getItem(k))]))')
        retired = {a['id'] for a in axes[:4]}
        assert len(saved['iota-studio-custom-axes']) == 1
        assert 'backup' not in saved[blocks_key]
        for p in [saved['iota-agent-state-v1']['profile'], saved['iota-studio-recipes'][0]['profile']]:
            assert not retired.intersection(p['coefficients'])
            assert not retired.intersection(a['id'] for a in p['custom_axes'])
            assert p['coefficients'][identity['id']] == 1.25
        with page.expect_download() as event:
            frame.locator('#exportProfile').click()
        exported = json.loads(Path(event.value.path()).read_text())
        assert next(a for a in exported['profile']['custom_axes'] if a.get('source')) == identity
        assert not retired.intersection(exported['profile']['coefficients'])
        page.screenshot(path=str(ROOT / 'runtime/cleanup-desktop.png'), full_page=True)
        page.reload()
        frame.locator('#blocksCanvas').wait_for(state='visible')
        assert frame.locator('.intervention-block').count() == 3
        page.set_viewport_size(dict(width=390, height=844))
        assert page.evaluate('document.documentElement.scrollWidth') <= 390
        assert page.evaluate("document.getElementById('scenariosFrame').contentDocument.documentElement.scrollWidth") <= 390
        assert not errors, errors
        print(json.dumps(dict(passed=True, checks=['legacy cache, draft and recipes cleaned; experimental preset and undo removed',
              'unrelated directions and strengths preserved', 'identity moved and relabeled without altering vector specification',
              'pixel cat avatar loaded and original intervention poles displayed', 'reload persistence and mobile layout'], page_errors=errors), ensure_ascii=False))
        browser.close()
    server.shutdown()


if __name__ == '__main__':
    main()
