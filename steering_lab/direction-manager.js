/* Delete browser-local custom directions, independently of removing active blocks. */
(() => {
  'use strict';
  const cacheKey = 'iota-studio-custom-axes';
  let undo = null;
  const notice = el('div', 'direction-delete-notice');
  notice.id = 'directionDeleteNotice';
  notice.setAttribute('role', 'status');
  notice.setAttribute('aria-live', 'polite');
  $('customSliders').after(notice);
  const help = el('p', 'mini muted', 'Delete direction removes it from your library and active combination. Remove intervention only disables its block.');
  $('customSliders').before(help);

  function message(text, canUndo = false) {
    notice.replaceChildren(el('span', '', text));
    if (canUndo) {
      const button = el('button', 'direction-undo', 'Undo');
      button.type = 'button';
      button.onclick = restore;
      notice.append(button);
    }
  }
  function payloadFor(c) {
    return load(cacheKey).find(r => r.axis?.id === c.id)
      || (D.direction_library || []).find(r => r.axis?.id === c.id)
      || {axis: clone(c.spec), native_direction: c.native_direction, direction: c.direction,
        points: c.points, diagnostics: c.diagnostics, positions: c.positions,
        provenance: {model: D.atlas.model, protocol_sha256: D.protocol_sha256}};
  }
  function remove(id) {
    const c = C.find(c => c.id === id);
    if (!c?.custom || c.source) return false;
    if (busy) { message('Wait for the current request to finish before deleting a direction.'); return false; }
    if (!retiredDirections.retire(id)) {
      message('Browser storage is unavailable. The direction was not deleted.');
      return false;
    }
    undo = {payload: clone(payloadFor(c)), coefficient: profile.coefficients[id] || 0,
      blocks: window.IOTABlocks?.snapshot(), name: c.name};
    persist(cacheKey, load(cacheKey).filter(r => r.axis?.id !== id));
    C.splice(C.indexOf(c), 1);
    delete profile.coefficients[id];
    profile.custom_axes = (profile.custom_axes || []).filter(a => a.id !== id);
    $('axis-' + id)?.closest('.axis')?.remove();
    for (const button of [...$('legend').children]) if (button.dataset.concept === id) button.remove();
    if (selected === id) selected = C[0].id;
    window.IOTABlocks?.restore(undo.blocks);
    syncInputs(); refresh();
    message('Direction deleted. Its active block was also removed.', true);
    $('buildDirection').disabled = busy;
    return true;
  }
  function restore() {
    if (!undo || busy) return;
    const saved = undo;
    if (C.filter(c => c.custom && !c.source).length >= 6) {
      message('The custom direction library is full. Delete another direction before undoing.', true);
      return;
    }
    if (!retiredDirections.restore(saved.payload.axis.id)) {
      message('Browser storage is unavailable. Could not restore the direction.', true);
      return;
    }
    addCustom(saved.payload);
    profile.coefficients[saved.payload.axis.id] = saved.coefficient;
    window.IOTABlocks?.restore(saved.blocks);
    syncInputs(); refresh(); undo = null;
    install();
    message('Direction restored.');
  }
  function connection() {
    for (const button of document.querySelectorAll('[data-delete-direction]')) button.disabled = busy;
  }
  function install() {
    for (const c of C.filter(c => c.custom && !c.source)) {
      const box = $('axis-' + c.id)?.closest('.axis');
      if (!box || box.querySelector('[data-delete-direction]')) continue;
      const button = el('button', 'delete-direction', 'Delete direction');
      button.type = 'button';
      button.dataset.deleteDirection = c.id;
      button.dataset.iotaNative = '';
      button.setAttribute('aria-label', 'Delete direction: ' + c.name);
      button.onclick = () => remove(c.id);
      button.disabled = busy;
      box.append(button);
    }
  }
  window.IOTADirections = {remove, restore, connection,
    isCustomLabel: value => C.some(c => c.custom && !c.source && [c.name, c.low, c.high].includes(value))};
  new MutationObserver(install).observe($('customSliders'), {childList: true});
  install();
})();
