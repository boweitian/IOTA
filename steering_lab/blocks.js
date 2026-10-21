/* A chain of editable interventions. Configuration, inference and human confirmation stay distinct. */
(() => {
  const key = 'iota-intervention-blocks-v1:' + D.protocol_sha256;
  let ready = false, order = [], confirmations = {}, tested = {}, draft = null, drawing = 0;
  const active = () => C.filter(c => Math.abs(profile.coefficients[c.id] || 0) > .001);
  const policy = () => D.strength_regularization || (D.total_strength_cap_enabled === false ? 'none' : 'norm');
  const signature = () => canon({profile: normalize(profile), regularization: policy()});
  const direction = id => C.find(c => c.id === id);
  function directionLabel(tag, cls, value, c) { const node = el(tag, cls, value); if (c?.custom && !c.source) node.dataset.iotaNative = ''; return node; }
  function persistBlocks() {
    persist(key, {order, confirmations, tested});
  }
  function ensureFaith() {
    if (!D.faith_preset) throw new Error('这个版本尚未打包身份认同实验方向。');
    if (!direction(D.faith_preset.axis.id)) addCustom(clone(D.faith_preset));
    return D.faith_preset.axis.id;
  }
  function initialize() {
    if (ready) return;
    try {
      const saved = JSON.parse(localStorage.getItem(key) || 'null');
      if (saved) {
        order = Array.isArray(saved.order) ? saved.order.filter(x => typeof x === 'string') : [];
        confirmations = saved.confirmations && typeof saved.confirmations === 'object' ? saved.confirmations : {};
        tested = saved.tested && typeof saved.tested === 'object' ? saved.tested : {};
      } else if (active().length) {
        $('blocksMigration').hidden = false;
        set('blocksMigration', '已把原来启用的 ' + active().length + ' 个方向转成积木，保留你的配置。点击“新建空白组合”可从零开始。');
      }
    } catch { order = []; }
    ready = true; refreshBlocks();
  }
  function refreshBlocks() {
    if (!ready) return;
    const ids = active().map(c => c.id);
    order = [...new Set(order.filter(id => ids.includes(id)).concat(ids))];
    const chain = $('blockChain'); chain.replaceChildren();
    const sig = signature(), measured = tested[sig] === true;
    for (const [index, id] of order.entries()) {
      const c = direction(id), value = profile.coefficients[id];
      const card = el('article', 'intervention-block' + (draft?.id === id ? ' selected' : ''));
      card.dataset.direction = id;
      const edit = el('button', 'block-edit'); edit.type = 'button'; edit.setAttribute('aria-label', '编辑干预：' + c.name);
      edit.append(el('span', 'block-number', String(index + 1).padStart(2, '0')), directionLabel('strong', '', c.name, c), directionLabel('span', 'block-pole', value >= 0 ? c.high : c.low, c));
      const strength = el('b', 'block-dose' + (Math.abs(value) > 1 ? ' elevated' : ''), Math.abs(value).toFixed(2) + '×');
      edit.append(strength, el('span', 'block-phase', confirmations[id] === sig ? '你已确认有效' : measured ? '已试聊 · 待你判断' : '已配置 · 待试聊'));
      if (c.source || c.positions === 'all') edit.append(el('small', 'block-experiment', '实验 · 全输入位置'));
      edit.onclick = () => open(id);
      const actions = el('div', 'block-actions'), extend = el('button', '', '接下一块'), remove = el('button', '', '删除干预');
      extend.type = remove.type = 'button'; extend.onclick = () => open(null, id); remove.onclick = () => removeBlock(id);
      actions.append(extend, remove); card.append(edit, actions); chain.append(card);
    }
    const ghost = el('button', 'block-ghost'); ghost.type = 'button'; ghost.id = 'addBlock';
    ghost.setAttribute('aria-label', order.length ? '添加下一个干预节点' : '添加第一个干预节点');
    ghost.append(el('span', 'ghost-circle', '+'), el('strong', '', order.length ? '下一块干预' : '第一块干预'), el('small', '', '点击虚线圆开始'));
    ghost.onclick = () => open(null); chain.append(ghost);
    chain.classList.toggle('empty', !order.length);
    $('blocksHighWarning').hidden = !active().some(c => Math.abs(profile.coefficients[c.id]) > 1);
    const mix = mixture();
    const strength = mix.regularization === 'direction_projection'
      ? '最大方向投影 ' + mix.max_projection.toFixed(2) + '× / 上限 ' + mix.projection_limit.toFixed(2) + '×' + (mix.cap_factor < 1 - 1e-9 ? ' · 全部系数乘以 ' + mix.cap_factor.toFixed(3) : '')
      : '组合强度 ' + (mix.norm / D.atlas.scale).toFixed(2) + '×' + (mix.regularization === 'none' ? ' · 自动缩放已暂停' : mix.cap_factor < .999 ? ' · 已整体缩放' : '');
    set('blocksSummary', order.length ? order.length + ' 个干预 · ' + strength : '空白组合 · 尚未施加干预');
    if (draft?.id && !order.includes(draft.id)) close();
    if (draft) renderInspector();
    persistBlocks(); scheduleDraw();
  }
  function options(selected) {
    const select = $('blockDirection'); select.replaceChildren();
    for (const [label, list] of [['内置方向', C.filter(c => !c.custom)], ['你的自定义方向', C.filter(c => c.custom && !c.source)], ['实验配置', C.filter(c => c.source)]]) {
      if (!list.length) continue;
      const group = document.createElement('optgroup'); group.label = label;
      for (const c of list) { const option = document.createElement('option'); option.value = c.id; option.textContent = c.name; if (c.custom && !c.source) option.dataset.iotaNative = ''; group.append(option); }
      select.append(group);
    }
    if (selected && direction(selected)) select.value = selected;
  }
  function open(id, after = null) {
    if (D.faith_preset) ensureFaith();
    const current = id ? profile.coefficients[id] : 1;
    draft = {id, direction: id || C.find(c => !order.includes(c.id))?.id || C[0].id, strength: Math.abs(current || 1), sign: current < 0 ? -1 : 1, after};
    $('blockInspector').hidden = false; $('blocksLayout').classList.add('inspector-open');
    options(draft.direction); renderInspector(); set('blockStatus', ''); refreshBlocks();
    $('blockDirection').focus();
  }
  function close() {
    draft = null; $('blockInspector').hidden = true; $('blocksLayout').classList.remove('inspector-open'); scheduleDraw();
  }
  function renderInspector() {
    if (!draft) return;
    const c = direction(draft.direction);
    set('blockEditorTitle', draft.id ? '编辑这块干预' : '添加一块干预');
    $('blockDirection').value = draft.direction;
    set('blockDirectionNote', c?.source || c?.positions === 'all' ? '实验方向：回答状态差，第 17 个 block，全输入位置注入。原实验为 2×；当前组合效果需重新试聊。' : c?.custom ? '使用你构建的内部方向，行为效果尚未认证。' : c?.note || '使用同一 Geometry 中的内置方向。');
    $('blockStrength').value = draft.strength;
    set('blockStrengthValue', draft.strength.toFixed(2) + '×');
    $('blockPositive').replaceChildren(document.createTextNode('正向 · '), directionLabel('span', '', c?.high || '', c)); $('blockNegative').replaceChildren(document.createTextNode('负向 · '), directionLabel('span', '', c?.low || '', c));
    $('blockPositive').setAttribute('aria-pressed', String(draft.sign > 0)); $('blockNegative').setAttribute('aria-pressed', String(draft.sign < 0));
    $('blockStrengthValue').classList.toggle('elevated', draft.strength > 1);
    $('applyBlock').textContent = draft.strength === 0 ? '停用这块干预' : draft.id ? '保存干预强度' : '应用干预，延伸下一块';
    $('deleteBlock').hidden = !draft.id;
    $('verifyBlock').hidden = !draft.id;
    $('confirmBlock').hidden = !draft.id || tested[signature()] !== true;
  }
  function apply() {
    if (!draft) return;
    if (!Number.isFinite(draft.strength) || draft.strength < 0 || draft.strength > 2) { set('blockStatus', '强度需要在 0 到 2× 之间。'); return; }
    const {id, direction: next, strength, sign, after} = draft;
    if (next !== id && order.includes(next)) { open(next); set('blockStatus', '这个方向已经有节点，已定位到它。'); return; }
    if (!strength) { if (id) removeBlock(id); else set('blockStatus', '0 表示不施加干预；调高强度后再应用。'); return; }
    if (id && id !== next) profile.coefficients[id] = 0;
    profile.coefficients[next] = strength * sign;
    if (id !== next) {
      const index = after ? order.indexOf(after) + 1 : id ? order.indexOf(id) : order.length;
      order = order.filter(x => x !== id && x !== next); order.splice(Math.max(0, index), 0, next);
    }
    delete confirmations[next]; draft.id = next;
    syncInputs(); refresh();
    set('blockStatus', '配置已应用，下一块虚线节点已延伸。行为效果需要试聊后判断。');
  }
  function removeBlock(id) {
    profile.coefficients[id] = 0; order = order.filter(x => x !== id); delete confirmations[id];
    if (draft?.id === id) close();
    syncInputs(); refresh();
  }
  function clear() {
    for (const c of C) profile.coefficients[c.id] = 0;
    order = []; confirmations = {}; tested = {}; close(); $('blocksMigration').hidden = true;
    syncInputs(); refresh();
  }
  function recordReply(result) {
    if (result.generation?.truncated) return;
    tested[canon({profile: normalize(result.profile), regularization: result.intervention.regularization || 'norm'})] = true;
    const entries = Object.keys(tested); if (entries.length > 30) for (const old of entries.slice(0, -30)) delete tested[old];
    refreshBlocks();
  }
  function scheduleDraw() { cancelAnimationFrame(drawing); drawing = requestAnimationFrame(draw); }
  function draw() {
    const canvas = $('blocksCanvas'), svg = $('blockConnections'); if (!canvas.clientWidth) return;
    const parent = canvas.getBoundingClientRect(); svg.replaceChildren(); svg.setAttribute('viewBox', '0 0 ' + canvas.clientWidth + ' ' + canvas.clientHeight);
    const nodes = [...$('blockChain').children];
    for (let i = 0; i + 1 < nodes.length; i++) {
      const a = nodes[i].getBoundingClientRect(), b = nodes[i + 1].getBoundingClientRect();
      const x1 = a.right - parent.left, y1 = a.top + a.height / 2 - parent.top;
      const x2 = b.left - parent.left, y2 = b.top + b.height / 2 - parent.top;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', Math.abs(y1 - y2) < 30 ? `M${x1} ${y1} C${x1 + 25} ${y1},${x2 - 25} ${y2},${x2} ${y2}` : `M${a.left + a.width / 2 - parent.left} ${a.bottom - parent.top} C${a.left + a.width / 2 - parent.left} ${a.bottom + 25 - parent.top},${b.left + b.width / 2 - parent.left} ${b.top - 25 - parent.top},${b.left + b.width / 2 - parent.left} ${b.top - parent.top}`);
      svg.append(path);
    }
  }
  window.IOTABlocks = {initialize, refresh: refreshBlocks, recordReply, snapshot: () => ({format: 'iota.blocks.v1', order: order.slice()}), restore: snapshot => { order = Array.isArray(snapshot?.order) ? snapshot.order.filter(id => typeof id === 'string' && direction(id)) : []; confirmations = {}; tested = {}; close(); refreshBlocks(); }, selectDirection: id => { if (ready) { open(null); draft.direction = id; options(id); renderInspector(); } }};
  $('createCustomBlock').onclick = () => { $('customAxes').open = true; $('customName').focus(); };
  $('closeBlockEditor').onclick = close; $('clearBlocks').onclick = clear;
  $('blockDirection').onchange = () => { if (!draft) return; draft.direction = $('blockDirection').value; renderInspector(); };
  $('blockStrength').oninput = () => { if (!draft) return; draft.strength = Number($('blockStrength').value); renderInspector(); };
  $('blockPositive').onclick = () => { if (draft) { draft.sign = 1; renderInspector(); } };
  $('blockNegative').onclick = () => { if (draft) { draft.sign = -1; renderInspector(); } };
  $('applyBlock').onclick = apply; $('deleteBlock').onclick = () => draft?.id && removeBlock(draft.id);
  $('verifyBlock').onclick = () => { window.IOTAAgent.show('chat'); $('chatInput').focus(); };
  $('confirmBlock').onclick = () => { if (draft?.id && tested[signature()] === true) { confirmations[draft.id] = signature(); refreshBlocks(); set('blockStatus', '已记录你的判断；这不是自动行为认证。'); } };
  new ResizeObserver(scheduleDraw).observe($('blocksCanvas')); addEventListener('resize', scheduleDraw);
})();
