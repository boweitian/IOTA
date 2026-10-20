/* One agent profile powers both the geometry editor and native multi-turn chat. */
(() => {
  const key = 'iota-agent-state-v1';
  const nameMigrationKey = 'iota-agent-name-xiaohei-v1';
  function renameLegacyDefault(p) {
    if (p.role.name !== '伊澜') return p;
    p.role.name = '小黑';
    for (const field of ['identity', 'boundaries']) p.role[field] = p.role[field].split('伊澜').join('小黑');
    p.role.facts = p.role.facts.map(fact => fact.split('伊澜').join('小黑'));
    return p;
  }
  // Keep archived experiment profiles intact; rename the editable agent only.
  renameLegacyDefault(profile);
  let transcript = [], sending = false, mode = 'chat', storageFailed = false;
  function storeState() {
    storageFailed = !persist(key, {binding: binding(), profile: clone(profile), messages: (transcript.length % 2 ? transcript.slice(0, -1) : transcript).slice(-200)});
  }
  function updateIdentity() {
    const name = profile.role.name.trim() || '未命名智能体';
    document.title = 'IOTA Search · ' + name;
    set('agentName', name); set('chatAgentName', name);
    const active = C.filter(c => Math.abs(profile.coefficients[c.id] || 0) > .001);
    set('agentSummary', active.length ? active.map(c => c.name).join(' · ') : '内在方向尚未施加');
    set('chatConfigNote', '下一条回复使用「' + name + '」当前的内在配置');
    storeState();
  }
  function connection() {
    $('sendChat').disabled = busy || !online;
    $('newConversation').disabled = sending;
    if (!sending) set('chatStatus', storageFailed ? '浏览器存储已满或不可用；本次对话暂存在当前页面。' : busy ? '模型正在处理内在配置，请稍候。' : online ? '支持连续对话 · Enter 发送，Shift + Enter 换行' : '模型尚未连接，连接后即可发送；可先塑造内在。');
  }
  function show(next, focus = false) {
    mode = next === 'inner' ? 'inner' : 'chat';
    for (const value of ['chat', 'inner']) {
      const stem = value === 'chat' ? 'agentChat' : 'agentInner', selected = value === mode;
      $(stem + 'Panel').hidden = !selected;
      $(stem + 'Tab').setAttribute('aria-selected', String(selected));
      $(stem + 'Tab').tabIndex = selected ? 0 : -1;
    }
    if (focus) $(mode === 'chat' ? 'agentChatTab' : 'agentInnerTab').focus();
    if (mode === 'inner') requestAnimationFrame(() => dispatchEvent(new Event('resize')));
  }
  function render() {
    const target = $('chatMessages'); target.replaceChildren();
    if (!transcript.length) {
      const empty = el('div', 'chat-empty');
      empty.append(el('strong', '', '从一句你好开始。'), el('p', '', '在「塑造内在」里设置名字、身份和行为，再来认识 TA。'));
      target.append(empty);
    }
    for (const message of transcript) {
      const article = el('article', 'chat-message ' + message.role);
      article.append(el('div', 'chat-speaker', message.role === 'user' ? '你' : message.agent_name || '智能体'), el('div', 'chat-text', message.content));
      if (message.role === 'assistant') {
        const note = message.truncated ? '回复达到长度上限，可继续追问。' : '使用发送时的内在配置';
        const details = el('details', 'chat-meta'); details.append(el('summary', '', note));
        details.append(el('p', '', '配置 ' + (message.profile_sha256 || '').slice(0, 12) + ' · 内部方向范数 ' + (message.norm ?? 0).toFixed(3)));
        const names = new Map([...C.map(c => [c.id, c.name]), ...(message.profile?.custom_axes || []).map(a => [a.id, a.source === D.faith_preset?.axis.source ? '身份认同' : a.name])]);
        const used = Object.entries(message.profile?.coefficients || {}).filter(([, v]) => Math.abs(v) > .001);
        const factor = message.cap_factor;
        const actualFactor = typeof factor === 'number' ? factor : 1;
        details.append(el('p', '', '实际注入的方向系数：' + (used.length ? used.map(([id, value]) => (names.get(id) || id) + ' ' + (value > 0 ? '+' : '') + (value * actualFactor).toFixed(2)).join('；') : '无（零干预）')));
        const intervention = message.intervention;
        if (intervention?.regularization === 'direction_projection') {
          details.append(el('p', '', '方向投影上限 ' + intervention.projection_limit.toFixed(2) + '× · 最大投影 ' + intervention.max_raw_projection.toFixed(2) + '× → ' + intervention.max_projection.toFixed(2) + '×。' + (actualFactor < 1 - 1e-9 ? '全部方向系数共同乘以 ' + actualFactor.toFixed(3) + ' 后注入。' : '按设置系数直接注入。')));
        } else if (typeof factor === 'number' && factor < .999) details.append(el('p', '', '达到总强度上限，全部方向系数共同乘以 ' + factor.toFixed(2) + ' 后注入。'));
        article.append(details);
      }
      target.append(article);
    }
    target.scrollTop = target.scrollHeight;
  }
  function contextFor(text) {
    const messages = [...transcript.map(m => ({role: m.role, content: m.content})), {role: 'user', content: text}];
    while (messages.length > 41 || messages.reduce((n, m) => n + m.content.length, 0) > 12000) messages.splice(0, 2);
    return messages;
  }
  async function send(event) {
    event.preventDefault();
    if (busy || sending) return;
    const text = $('chatInput').value.trim();
    if (!text) { set('chatStatus', '先写一句想说的话。'); return; }
    if (text.length > 2000) { set('chatStatus', '一条消息最多 2000 字。'); return; }
    if (!online) { connection(); return; }
    let snapshot;
    try { snapshot = verify(profile); } catch (error) { set('chatStatus', error.message); return; }
    const messages = contextFor(text), retained = messages.length;
    transcript.push({role: 'user', content: text});
    sending = true; setBusy(true); render(); $('chatInput').value = ''; $('chatInput').readOnly = true;
    set('chatStatus', snapshot.role.name + ' 正在回复…');
    try {
      const result = await api('/api/studio/chat', {profile: snapshot, messages, binding: binding()});
      if (result.message?.role !== 'assistant' || typeof result.message.content !== 'string' || !result.message.content.trim()) throw new Error('模型未返回有效回复，请重试。');
      transcript.push({...result.message, agent_name: result.agent_name, profile_sha256: result.profile_sha256, norm: result.intervention.norm, cap_factor: result.intervention.cap_factor, intervention: result.intervention, truncated: result.generation.truncated, profile: result.profile});
      storeState(); render(); window.IOTABlocks?.recordReply(result);
      set('chatStatus', storageFailed ? '浏览器存储不可用；本次对话暂存在当前页面。' : (canon(snapshot) !== canon(normalize(profile)) ? '这条回复使用发送时的配置；下一条使用你修改后的内在。' : '已回复。') + (retained < transcript.length - 1 ? ' 本次使用最近的对话上下文。' : ''));
    } catch (error) {
      transcript.pop(); $('chatInput').value = text; render();
      set('chatStatus', '发送未完成：' + error.message + ' 消息已放回输入框，可重试。');
    } finally {
      setBusy(false); sending = false; $('chatInput').readOnly = false;
      $('sendChat').disabled = busy || !online; $('newConversation').disabled = false;
      $('chatInput').focus();
    }
  }
  window.IOTAAgent = {refresh: updateIdentity, connection, show};
  let restoredDraft = false;
  try {
    const saved = JSON.parse(localStorage.getItem(key) || 'null');
    if (saved && canon(saved.binding) === canon(binding())) {
      profile = verify(saved.profile);
      if (!localStorage.getItem(nameMigrationKey)) renameLegacyDefault(profile);
      presetId = ''; restoredDraft = true; syncInputs(); refresh();
      if (Array.isArray(saved.messages) && saved.messages.length % 2 === 0 && saved.messages.every((m, i) => m?.role === (i % 2 ? 'assistant' : 'user') && typeof m.content === 'string' && m.content.trim() && m.content.length <= 4000)) transcript = saved.messages.slice(-200);
    }
  } catch { /* An invalid draft must not prevent starting a fresh agent. */ }
  if (!restoredDraft) { for (const c of C) profile.coefficients[c.id] = 0; syncInputs(); refresh(); }
  persist(nameMigrationKey, true);
  $('agentChatTab').onclick = () => show('chat'); $('agentInnerTab').onclick = () => show('inner');
  for (const id of ['agentChatTab', 'agentInnerTab']) $(id).onkeydown = event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault(); show(event.key === 'Home' ? 'chat' : event.key === 'End' ? 'inner' : mode === 'chat' ? 'inner' : 'chat', true);
  };
  $('chatForm').onsubmit = send;
  $('chatInput').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('chatForm').requestSubmit(); } };
  $('newConversation').onclick = () => { if (sending) return; transcript = []; storeState(); render(); connection(); };
  let initialView = 'chat';
  try { if (new URLSearchParams(location.search).get('view') === 'inner' || (parent !== window && new URLSearchParams(parent.location.search).get('view') === 'inner')) initialView = 'inner'; } catch {}
  updateIdentity(); render(); show(initialView); connection(); window.IOTABlocks?.initialize();
})();
