/* English presentation for the existing IOTA UI. Model data and requests stay native. */
(() => {
  'use strict';
  const catalog = __IOTA_EN_CATALOG__;
  const escape = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(Object.keys(catalog).sort((a, b) => b.length - a.length).map(escape).join('|'), 'g');
  const translate = value => typeof value === 'string' ? value.replace(pattern, match => catalog[match]) : value;
  // Never translate evidence, conversation content, editable fields, or embedded data.
  const native = 'script,style,textarea,input,pre,.chat-text,.piecebody,[data-iota-native],#premise,#hypothesis,#reason,#context,#manualObservation,#annotation';
  const attributes = ['title', 'aria-label', 'alt', 'placeholder', 'label'];
  function text(node) {
    if (!node.parentElement || node.parentElement.closest(native)) return;
    const original = node.nodeValue;
    if (!/[\u3400-\u9fff]/.test(original)) return;
    // Option labels may be translated, but their submitted values must remain exact.
    const option = node.parentElement.closest('option');
    if (option && !option.hasAttribute('value')) option.setAttribute('value', option.value);
    const english = node.parentElement.matches('.agent-chat .sectionhead h2') && original.trim() === '聊天' ? '' : translate(original);
    if (english !== original) node.nodeValue = english;
  }
  function element(node) {
    if (node.matches('script,style,[data-iota-native]')) return;
    for (const name of attributes) {
      const value = node.getAttribute(name);
      if (value) {
        const english = translate(value);
        if (english !== value) node.setAttribute(name, english);
      }
    }
    if (node.matches('meta[name="description"]')) {
      const value = node.getAttribute('content');
      const english = translate(value);
      if (english !== value) node.setAttribute('content', english);
    }
    for (const child of node.childNodes) visit(child);
  }
  function visit(node) {
    if (node.nodeType === Node.TEXT_NODE) text(node);
    else if (node.nodeType === Node.ELEMENT_NODE) element(node);
  }
  // Canvas labels use the same catalog, with matching text metrics for positioning.
  if (window.CanvasRenderingContext2D) {
    for (const method of ['fillText', 'strokeText', 'measureText']) {
      const original = CanvasRenderingContext2D.prototype[method];
      CanvasRenderingContext2D.prototype[method] = function (value, ...args) {
        return Reflect.apply(original, this, [window.IOTADirections?.isCustomLabel(value) ? value : translate(value), ...args]);
      };
    }
  }
  function start() {
    document.documentElement.lang = 'en';
    visit(document.documentElement);
    new MutationObserver(records => {
      for (const record of records) {
        if (record.type === 'characterData') text(record.target);
        else if (record.type === 'attributes') element(record.target);
        else for (const node of record.addedNodes) visit(node);
      }
    }).observe(document.documentElement, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: [...attributes, 'content']});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
  else start();
  window.IOTAEnglish = Object.freeze({translate, language: 'en'});
})();
