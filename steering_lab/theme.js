/* Run before paint; synchronize the preference across same-origin modules. */
(() => {
  'use strict';
  if (window.IOTATheme) return;
  const key = 'iota:theme:v1', valid = value => value === 'green' || value === 'blue';
  let current = 'green';
  try { const saved = localStorage.getItem(key); if (valid(saved)) current = saved; } catch {}
  function apply(value, {persist = false, notify = false} = {}) {
    current = valid(value) ? value : 'green';
    document.documentElement.dataset.theme = current;
    document.querySelectorAll('#themeSelect').forEach(select => { select.value = current; });
    if (persist) { try { localStorage.setItem(key, current); } catch {} }
    if (notify) document.querySelectorAll('iframe').forEach(frame => {
      frame.contentWindow?.postMessage({type: 'iota:theme', theme: current}, location.origin);
    });
  }
  window.IOTATheme = {get: () => current, set: value => apply(value, {persist: true, notify: true})};
  apply(current);
  addEventListener('storage', event => {
    if (event.key === key || event.key === null) apply(event.newValue, {notify: true});
  });
  addEventListener('message', event => {
    if (event.origin !== location.origin || event.source !== parent || parent === window) return;
    if (event.data?.type === 'iota:theme' && valid(event.data.theme)) apply(event.data.theme);
  });
  addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('#themeSelect').forEach(select => {
      select.value = current;
      select.addEventListener('change', () => window.IOTATheme.set(select.value));
    });
    document.querySelectorAll('iframe').forEach(frame => {
      frame.addEventListener('load', () => frame.contentWindow?.postMessage({type: 'iota:theme', theme: current}, location.origin));
    });
  });
})();
