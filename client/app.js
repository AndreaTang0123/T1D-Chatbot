// xiaozhi web-client —— 文字对话 + 按住说话 + 语音回复（TTS 由 tts-player.js 在浏览器端完成）
// 协议细节见 PROTOCOL.md

'use strict';

// ---------- 身份 ----------

function makeUuid() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

// localStorage 可能不可用（隐私模式），失败时每次生成新的
function loadOrCreate(key, factory) {
  try {
    let v = localStorage.getItem(key);
    if (!v) {
      v = factory();
      localStorage.setItem(key, v);
    }
    return v;
  } catch (e) {
    return factory();
  }
}

// server 按 client-id 加载人设和数据：data/<client-id>/{prompt.txt,prompts/,config.json,memory.json}
// 默认用 Joe（server/data/<Joe 的 client-id>/ 下的人设、记忆和 CGM/pump 数据）；测试或换人设时用 ?client-id=<id> 覆盖
const JOE_CLIENT_ID = '26ea0ba9-2d55-4368-a56d-19c4a27c0772';

const deviceId = loadOrCreate('xz-device-id', () => 'web-' + makeUuid().slice(0, 8));
const clientId = new URLSearchParams(location.search).get('client-id') || JOE_CLIENT_ID;

// 默认同源；本地调试可用 ?ws=ws://127.0.0.1:8000/xiaozhi/v1/ 覆盖
function buildWsUrl() {
  const override = new URLSearchParams(location.search).get('ws');
  const base = override || `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/xiaozhi/v1/`;
  const url = new URL(base);
  url.searchParams.set('device-id', deviceId);
  url.searchParams.set('client-id', clientId);
  return url.toString();
}

// ---------- UI ----------

const $messages = document.getElementById('messages');
const $form = document.getElementById('inputForm');
const $input = document.getElementById('textInput');
const $send = document.getElementById('sendBtn');
const $dot = document.getElementById('statusDot');
const $statusText = document.getElementById('statusText');
const $reconnect = document.getElementById('reconnectBtn');
const $phase = document.getElementById('phase');
const $starters = document.getElementById('starters');

// state: 'disconnected' | 'connecting' | 'ready'
function setStatus(state, text) {
  $dot.className = state === 'disconnected' ? '' : state;
  $statusText.textContent = text;
  $send.disabled = state !== 'ready';
  $reconnect.hidden = state !== 'disconnected';
  document.querySelectorAll('.ask').forEach((b) => { b.disabled = state !== 'ready'; });
  updateTalkButton();
}

function scrollToBottom() {
  $messages.scrollTop = $messages.scrollHeight;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clockTime(date) {
  return (date || new Date()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// 一条消息 = row > [头像] + col > bubble + meta（时间）+ 可选的数据依据
function addBubble(role, text) {
  const row = el('div', 'row ' + role);
  if (role === 'bot') row.appendChild(el('div', 'avatar small', 'J'));
  const col = el('div', 'col');
  const bubble = el('div', 'bubble ' + role);
  renderRich(bubble, text);
  col.appendChild(bubble);
  col.appendChild(el('div', 'meta', clockTime()));
  row.appendChild(col);
  $messages.appendChild(row);
  if (role === 'user') $starters.remove();
  scrollToBottom();
  return bubble;
}

function addNotice(text) {
  $messages.appendChild(el('div', 'notice', text));
  scrollToBottom();
}

// ---------- 数值高亮 ----------
// 回复里的 "143 mg/dL"、"72%"、"1.5 U" 做成小标签；血糖值按区间上色。
// 只生成文本节点和 span，不用 innerHTML（回复来自 LLM）。

const VALUE_RE = /(\d+(?:\.\d+)?)\s?(mg\/dL|mmol\/L|%|units?\b|U\b)/gi;
// "decreased by 38 mg/dL"、"a drop of 40 mg/dL"、"+25 mg/dL" 是变化量，不是血糖值，不按区间上色
const DELTA_BEFORE_RE = /(\bby|\bof|[+\-−±])\s*$/i;

// mg/dL：<70 低，70-180 目标内，181-250 偏高，>250 很高
function glucoseClass(mgdl) {
  if (mgdl == null || isNaN(mgdl)) return 'g-none';
  if (mgdl < 70) return 'g-low';
  if (mgdl <= 180) return 'g-ok';
  if (mgdl <= 250) return 'g-high';
  return 'g-vhigh';
}

function renderRich(node, text) {
  node.replaceChildren();
  let last = 0;
  for (const m of text.matchAll(VALUE_RE)) {
    if (m.index > last) node.appendChild(document.createTextNode(text.slice(last, m.index)));
    const chip = el('span', 'chip', m[0]);
    const unit = m[2].toLowerCase();
    const isDelta = DELTA_BEFORE_RE.test(text.slice(Math.max(0, m.index - 12), m.index));
    if (isDelta) { /* 变化量：保持中性标签 */ }
    else if (unit === 'mg/dl') chip.classList.add(glucoseClass(parseFloat(m[1])));
    else if (unit === 'mmol/l') chip.classList.add(glucoseClass(parseFloat(m[1]) * 18));
    node.appendChild(chip);
    last = m.index + m[0].length;
  }
  if (last < text.length) node.appendChild(document.createTextNode(text.slice(last)));
}

// ---------- 思考中 ----------
// stt（用户这句话）到达后显示，第一句回复或 tts stop 时移除；
// get_diabetes_data 运行时 server 推送 coach step，标签随之变化。

const STEP_LABELS = {
  data: 'Syncing your CGM & pump data…',
  decision: 'Looking for patterns…',
  interpretation: 'Putting it into words…',
};

let typingRow = null;
let waitingReply = false;  // 已收到用户这句话（stt），回复还没开始
let transcribing = false;  // 已松开按钮，还没收到 stt
let transcribeTimer = null;

function showTyping(label) {
  waitingReply = true;
  if (!typingRow) {
    typingRow = el('div', 'row bot');
    typingRow.appendChild(el('div', 'avatar small', 'J'));
    const bubble = el('div', 'bubble bot typing');
    const dots = el('span', 'dots');
    dots.append(el('i'), el('i'), el('i'));
    bubble.append(dots, el('span', 'typing-label'));
    typingRow.appendChild(bubble);
  }
  typingRow.querySelector('.typing-label').textContent = label || 'Thinking…';
  $messages.appendChild(typingRow); // 已在页面里时会移到最后
  scrollToBottom();
}

function hideTyping() {
  waitingReply = false;
  if (typingRow) typingRow.remove();
  typingRow = null;
}

function setTranscribing(on) {
  transcribing = on;
  clearTimeout(transcribeTimer);
  // 没识别出文字时 server 什么都不回，10s 后放弃等待
  if (on) transcribeTimer = setTimeout(() => { transcribing = false; }, 10000);
}

// ---------- 助手回复合并 ----------
// server 每句发一条 tts sentence_start（去过 markdown），
// LLM 结束时再发一条带全文的 sentence_start（原文去 emoji）。
// 比较时去掉空白、标点和符号，避免 markdown 差异造成重复显示。

function normalize(s) {
  return s.replace(/[\s\p{P}\p{S}]/gu, '');
}

let botBubble = null; // 本轮的助手气泡
let botText = '';
let pendingEvidence = null; // 本轮 get_diabetes_data 的数据依据，等气泡出现后挂上去

function resetTurn() {
  botBubble = null;
  botText = '';
  pendingEvidence = null;
}

function onAssistantText(text) {
  const incoming = normalize(text);
  if (!incoming) return;
  const current = normalize(botText);

  if (current.includes(incoming)) {
    console.log('[ws] sentence_start already shown, skip');
    return;
  }
  if (incoming.includes(current)) {
    // 全文（或更长的版本）到达，整体替换
    botText = text;
  } else {
    botText = botText ? botText + ' ' + text : text;
  }

  hideTyping();
  if (!botBubble) botBubble = addBubble('bot', botText);
  else renderRich(botBubble, botText);
  attachEvidence();
  scrollToBottom();
}

function attachEvidence() {
  if (!pendingEvidence || !botBubble) return;
  botBubble.parentElement.appendChild(renderEvidence(pendingEvidence));
  pendingEvidence = null;
}

// ---------- 数据依据 ----------
// 只展示本次真正交给模型的数据（server 的 get_diabetes_data → build_evidence）。

const DIRECTION_ARROWS = {
  DoubleUp: '⇈', SingleUp: '↑', FortyFiveUp: '↗', Flat: '→',
  FortyFiveDown: '↘', SingleDown: '↓', DoubleDown: '⇊',
};
const DIRECTION_WORDS = {
  DoubleUp: 'rising fast', SingleUp: 'rising', FortyFiveUp: 'rising slowly', Flat: 'steady',
  FortyFiveDown: 'falling slowly', SingleDown: 'falling', DoubleDown: 'falling fast',
};

function cgmBadge(status, ageMin) {
  if (status === 'ok') return el('span', 'badge ok', 'CGM live' + (ageMin != null ? ' · ' + ageMin + ' min' : ''));
  if (status === 'stale') return el('span', 'badge warn', 'CGM ' + formatAge(ageMin * 60) + ' old');
  return el('span', 'badge bad', 'No CGM data');
}

function pumpBadge(ds) {
  if (!ds || !ds.status || ds.status === 'no_data') return el('span', 'badge', 'No pump data');
  if (ds.status === 'ok') return el('span', 'badge ok', 'Pump synced');
  return el('span', 'badge warn', 'Pump ' + formatAge(ds.age_hours * 3600) + ' old');
}

const OVERALL_CLASSES = { stable: 'ok', 'needs attention': 'warn', 'high risk': 'bad' };

function renderEvidence(ev) {
  const details = el('details', 'evidence');
  const summary = el('summary');
  summary.appendChild(el('span', null, 'Data behind this answer'));
  const decision = ev.decision || {};
  if (decision.overall_status) {
    summary.appendChild(el('span', 'badge ' + (OVERALL_CLASSES[decision.overall_status] || ''), decision.overall_status));
  }
  summary.appendChild(cgmBadge(ev.cgm.status, ev.cgm.age_min));
  if (ev.topic !== 'glucose') summary.appendChild(pumpBadge(ev.pump.data_status));
  details.appendChild(summary);

  const body = el('div', 'ev-body');
  const section = (title) => body.appendChild(el('h4', null, title));
  const list = (items) => {
    const ul = el('ul');
    for (const item of items) {
      const li = el('li');
      renderRich(li, String(item));
      ul.appendChild(li);
    }
    body.appendChild(ul);
  };
  const para = (text, cls) => {
    const p = el('p', cls);
    renderRich(p, text);
    body.appendChild(p);
  };

  if (decision.focus_message || (decision.primary_issues || []).length) {
    section("Joe's read");
    if (decision.focus_message) para(decision.focus_message, 'ev-focus');
    const issues = [].concat(decision.primary_issues || [], decision.secondary_issues || []);
    if (issues.length) list(issues);
    const pats = [].concat(decision.patterns || [], decision.pump_context || []);
    if (pats.length) list(pats);
  }

  section('Glucose');
  const cgm = ev.cgm;
  if (cgm.status === 'ok') {
    para(cgm.glucose_now + ' mg/dL, ' + (DIRECTION_WORDS[cgm.direction] || cgm.direction || '') + ', ' + cgm.age_min + ' min ago');
  } else if (cgm.status === 'stale') {
    para('No current reading. Last was ' + cgm.last_glucose + ' mg/dL, ' + formatAge(cgm.age_min * 60) + ' ago.');
  } else {
    para('No CGM reading available.');
  }
  // patterns.daily 是 "[CGM Data]\n当前读数\n规律 + TIR" 这样的文本块，去掉标题和重复的当前读数行
  const lines = [];
  for (const block of cgm.context || []) {
    for (const line of String(block).split('\n')) {
      const t = line.trim();
      if (t && !/^\[CGM Data\]$/.test(t) && !/^Current Glucose:/.test(t)) lines.push(t);
    }
  }
  const metrics = Object.entries(cgm.metrics || {}).map(([k, v]) => k + ': ' + (typeof v === 'object' ? JSON.stringify(v) : v));
  if (lines.length || metrics.length) list(lines.concat(metrics));

  const pump = ev.pump;
  const pumpItems = [].concat(pump.patterns || [], pump.insulin_effect || [], pump.anomalies || []);
  if (ev.topic !== 'glucose' && (pumpItems.length || pump.summary)) {
    section('Pump & insulin');
    if (pumpItems.length) list(pumpItems);
    if (pump.summary) {
      const raw = el('details');
      raw.appendChild(el('summary', null, 'Raw pump summary'));
      raw.appendChild(el('p', 'ev-raw', pump.summary));
      body.appendChild(raw);
    }
  }

  details.appendChild(body);
  details.addEventListener('toggle', () => { if (details.open) details.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); });
  return details;
}

// ---------- 血糖胶囊 + Today 面板 ----------
// 数据来自 {"type":"status"} 的回复（server/core/utils/status_snapshot.py）。
// 连上后、每分钟、页面回到前台、每次数据分析后各请求一次；读数年龄在本地按收到时间往后推。

const STATUS_POLL_MS = 60000;
const $pill = document.getElementById('glucosePill');
const $sheet = document.getElementById('todaySheet');
const $backdrop = document.getElementById('sheetBackdrop');
const $todayBody = document.getElementById('todayBody');

let lastStatus = null;
let statusReceivedAt = 0;

function requestStatus() {
  if (sessionId) sendJson({ type: 'status' });
}

function onStatus(msg) {
  lastStatus = msg;
  statusReceivedAt = Date.now();
  renderPill();
  if (!$sheet.hidden) renderToday();
}

// 秒 → "3 min" / "2 h 5 min" / "3 d"
function formatAge(seconds) {
  const min = Math.max(0, Math.round(seconds / 60));
  if (min < 60) return min + ' min';
  const h = Math.floor(min / 60);
  if (h < 48) return h + ' h' + (min % 60 && h < 6 ? ' ' + (min % 60) + ' min' : '');
  return Math.floor(h / 24) + ' d';
}

// 当前 CGM 读数的实时状态（年龄按收到快照后经过的时间推算）
function liveCgm() {
  const cgm = lastStatus && lastStatus.cgm;
  if (!cgm || !cgm.enabled) return null;
  if (cgm.glucose == null || cgm.age_min == null) return { cgm, state: 'none' };
  const ageMin = cgm.age_min + Math.floor((Date.now() - statusReceivedAt) / 60000);
  const stale = ageMin > (cgm.stale_after_min || 15);
  return { cgm, ageMin, state: stale ? 'stale' : 'ok' };
}

function renderPill() {
  const live = liveCgm();
  $pill.hidden = !live;
  if (!live) return;
  const value = $pill.querySelector('.pill-value');
  const arrow = $pill.querySelector('.pill-arrow');
  const age = $pill.querySelector('.pill-age');
  if (live.state === 'none') {
    $pill.className = 'g-none';
    value.textContent = '--';
    arrow.textContent = '';
    age.textContent = 'No data';
    $pill.setAttribute('aria-label', "No CGM reading. Open today's summary");
    return;
  }
  const { cgm, ageMin } = live;
  $pill.className = live.state === 'stale' ? 'g-stale' : glucoseClass(cgm.glucose);
  value.textContent = cgm.glucose;
  arrow.textContent = live.state === 'stale' ? '' : (DIRECTION_ARROWS[cgm.direction] || '');
  age.textContent = ageMin < 1 ? 'just now' : formatAge(ageMin * 60) + ' ago';
  $pill.setAttribute('aria-label', 'Glucose ' + cgm.glucose + ' mg/dL, ' +
    (DIRECTION_WORDS[cgm.direction] || '') + ', ' + age.textContent + ". Open today's summary");
}

// 目标值来自国际 CGM 共识（Battelino 2019）
function tile(label, value, unit, goal, good) {
  const t = el('div', 'tile' + (good == null ? '' : good ? ' good' : ' bad'));
  t.appendChild(el('div', 'tile-label', label));
  const v = el('div', 'tile-value', value == null ? '--' : String(value));
  if (unit && value != null) v.appendChild(el('small', null, unit));
  t.appendChild(v);
  if (goal) t.appendChild(el('div', 'tile-goal', goal));
  return t;
}

function rangeTiles(s, extra) {
  const tiles = el('div', 'tiles');
  tiles.append(
    tile('In range 70–180', s.tir, '%', 'goal > 70%', s.tir >= 70),
    tile('Average', s.mean, 'mg/dL', null, null),
    tile('Below 70', s.tbr, '%', 'goal < 4%', s.tbr < 4),
    tile('Above 180', s.tar, '%', 'goal < 25%', s.tar < 25),
  );
  if (extra) tiles.append(...extra);
  return tiles;
}

function listRow(label, value) {
  const row = el('div', 'list-row');
  row.append(el('span', null, label), el('span', null, value));
  return row;
}

const QUICK_ASKS = [
  'How has my glucose been today?',
  'Any patterns in the last two weeks?',
  'Did my last bolus work?',
];

function askButtons(className) {
  const wrap = el('div', className);
  for (const q of QUICK_ASKS) {
    const b = el('button', 'ask', q);
    b.type = 'button';
    b.disabled = !sessionId;
    b.addEventListener('click', () => {
      unlockAudio();
      if (sendText(q)) closeToday();
    });
    wrap.appendChild(b);
  }
  return wrap;
}

function renderToday() {
  const body = $todayBody;
  body.replaceChildren();
  if (!lastStatus) {
    body.appendChild(el('p', 'empty', sessionId ? 'Loading…' : 'Not connected.'));
    return;
  }

  const live = liveCgm();
  if (!live) {
    body.appendChild(el('p', 'empty', 'No CGM is connected for this account.'));
  } else {
    const now = el('div', 'now ' + (live.state === 'ok' ? glucoseClass(live.cgm.glucose) : 'g-stale'));
    const value = el('div', 'now-value', live.state === 'none' ? '--' : String(live.cgm.glucose));
    if (live.state === 'ok') value.appendChild(el('span', 'arrow', DIRECTION_ARROWS[live.cgm.direction] || ''));
    now.appendChild(value);
    let sub = 'No reading available';
    if (live.state === 'ok') {
      sub = 'mg/dL · ' + (DIRECTION_WORDS[live.cgm.direction] || live.cgm.direction) + ' · ' +
        (live.ageMin < 1 ? 'just now' : formatAge(live.ageMin * 60) + ' ago');
    } else if (live.state === 'stale') {
      sub = 'Last reading ' + formatAge(live.ageMin * 60) + ' ago, not current';
    }
    now.appendChild(el('div', 'now-sub', sub));
    body.appendChild(now);

    const today = live.cgm.today;
    body.appendChild(el('h3', null, 'Today so far'));
    if (today) {
      body.appendChild(rangeTiles(today));
      body.appendChild(el('p', 'foot', 'Lowest ' + today.min + ' · Highest ' + today.max + ' · ' + today.n + ' readings'));
    } else {
      body.appendChild(el('p', 'empty', 'No readings since midnight.'));
    }

    const d14 = live.cgm.days14;
    const h14 = el('h3', null, 'Last 14 days');
    if (d14) h14.appendChild(el('small', null, d14.days + (d14.days === 1 ? ' day' : ' days') + ' of data'));
    body.appendChild(h14);
    if (d14) {
      body.appendChild(rangeTiles(d14, [
        tile('GMI', d14.gmi, '%', 'goal < 7%', d14.gmi < 7),
        tile('Variability (CV)', d14.cv, '%', 'goal ≤ 36%', d14.cv == null ? null : d14.cv <= 36),
      ]));
    } else {
      body.appendChild(el('p', 'empty', 'No readings in the last 14 days.'));
    }
  }

  const pump = lastStatus.pump;
  if (pump && pump.enabled) {
    body.appendChild(el('h3', null, 'Pump'));
    const nowS = lastStatus.server_time + (Date.now() - statusReceivedAt) / 1000;
    const b = pump.last_bolus;
    if (b) {
      let text = (b.insulin_units != null ? b.insulin_units + ' U' : 'Bolus');
      if (b.carbs_g) text += ' · ' + b.carbs_g + ' g carbs';
      text += ' · ' + formatAge(nowS - b.unix_s) + ' ago';
      body.appendChild(listRow('Last bolus', text));
    }
    const sync = el('span');
    sync.appendChild(pumpBadge(pump));
    const row = el('div', 'list-row');
    row.append(el('span', null, 'Pump data'), sync);
    body.appendChild(row);
  }

  body.appendChild(el('h3', null, 'Ask Joe'));
  body.appendChild(askButtons('asks'));
  body.appendChild(el('p', 'foot', 'Updated ' + clockTime(new Date(statusReceivedAt)) + ' · not medical advice'));
}

function openToday() {
  renderToday();
  requestStatus();
  $backdrop.hidden = false;
  $sheet.hidden = false;
  // 先显示再加 class，transition 才会生效
  requestAnimationFrame(() => {
    $backdrop.classList.add('open');
    $sheet.classList.add('open');
  });
  document.getElementById('sheetClose').focus();
}

function closeToday() {
  if ($sheet.hidden) return;
  $backdrop.classList.remove('open');
  $sheet.classList.remove('open');
  setTimeout(() => {
    $backdrop.hidden = true;
    $sheet.hidden = true;
  }, 220);
  $pill.focus();
}

$pill.addEventListener('click', openToday);
$backdrop.addEventListener('click', closeToday);
document.getElementById('sheetClose').addEventListener('click', closeToday);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeToday(); });

// ---------- Opus 解码 ----------
// libopus.js 是 asm.js 构建，堆固定 16MB 且不可增长，所以输入/输出缓冲只分配一次。
// server 下发：16kHz、单声道、60ms（960 采样点）一包。

const OPUS_RATE = 16000;
const OPUS_MAX_FRAME = 1920; // 16kHz 下 opus 单包最长 120ms
const OPUS_MAX_PACKET = 4000;

function createOpusDecoder() {
  const mod = (typeof Module !== 'undefined' && Module.instance) || null;
  if (!mod || typeof mod._opus_decoder_get_size !== 'function') {
    console.log('[audio] libopus not loaded, Module.instance missing');
    return null;
  }

  const size = mod._opus_decoder_get_size(1);
  const decPtr = mod._malloc(size);
  const inPtr = mod._malloc(OPUS_MAX_PACKET);
  const outPtr = mod._malloc(OPUS_MAX_FRAME * 2);

  function reset() {
    const err = mod._opus_decoder_init(decPtr, OPUS_RATE, 1);
    if (err < 0) console.log('[audio] opus_decoder_init failed', err);
    return err >= 0;
  }

  // 返回 Float32Array（[-1,1)），失败返回 null
  function decode(packet) {
    if (packet.length === 0 || packet.length > OPUS_MAX_PACKET) {
      console.log('[audio] skip packet, length=' + packet.length);
      return null;
    }
    mod.HEAPU8.set(packet, inPtr);
    const n = mod._opus_decode(decPtr, inPtr, packet.length, outPtr, OPUS_MAX_FRAME, 0);
    if (n < 0) {
      console.log('[audio] opus_decode error', n);
      return null;
    }
    const pcm = mod.HEAP16.subarray(outPtr >> 1, (outPtr >> 1) + n);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = pcm[i] / 32768;
    return out;
  }

  if (!reset()) return null;
  console.log('[audio] opus decoder ready, state size=' + size);
  return { decode, reset };
}

// ---------- Opus 编码 ----------
// 上行：16kHz、单声道、每帧 960 个 Int16 采样点。
// 注意 _opus_encoder_ctl 是变参函数，asm.js 里变参要以「指向堆内存的指针」传入；
// 官方测试页直接传数值，实测会把码率设成 300000。

const OPUS_APPLICATION_VOIP = 2048;
const OPUS_SET_BITRATE = 4002;
const OPUS_BITRATE = 24000;

function createOpusEncoder() {
  const mod = (typeof Module !== 'undefined' && Module.instance) || null;
  if (!mod || typeof mod._opus_encoder_get_size !== 'function') {
    console.log('[mic] libopus not loaded, Module.instance missing');
    return null;
  }

  const encPtr = mod._malloc(mod._opus_encoder_get_size(1));
  const pcmPtr = mod._malloc(960 * 2);
  const outPtr = mod._malloc(OPUS_MAX_PACKET);
  const argPtr = mod._malloc(4);

  function reset() {
    const err = mod._opus_encoder_init(encPtr, OPUS_RATE, 1, OPUS_APPLICATION_VOIP);
    if (err < 0) {
      console.log('[mic] opus_encoder_init failed', err);
      return false;
    }
    mod.HEAP32[argPtr >> 2] = OPUS_BITRATE;
    const ret = mod._opus_encoder_ctl(encPtr, OPUS_SET_BITRATE, argPtr);
    if (ret < 0) console.log('[mic] OPUS_SET_BITRATE failed', ret);
    return true;
  }

  // pcm: Int16Array(960)；返回 Uint8Array，失败返回 null
  function encode(pcm) {
    mod.HEAP16.set(pcm, pcmPtr >> 1);
    const len = mod._opus_encode(encPtr, pcmPtr, pcm.length, outPtr, OPUS_MAX_PACKET);
    if (len < 0) {
      console.log('[mic] opus_encode error', len);
      return null;
    }
    return mod.HEAPU8.slice(outPtr, outPtr + len);
  }

  if (!reset()) return null;
  console.log('[mic] opus encoder ready, bitrate=' + OPUS_BITRATE);
  return { encode, reset };
}

// ---------- 重采样 ----------
// 16kHz → AudioContext 原生采样率的流式线性插值。
// 跨包保留上一包最后一个采样点和小数位置，保证包与包之间波形连续（无爆音）。

function createResampler(inRate, outRate) {
  const step = inRate / outRate;
  let last = 0; // 上一包最后一个采样点，起始为静音
  let pos = 0;  // 在 [last, ...input] 这个扩展序列里的位置

  function process(input) {
    if (inRate === outRate) return input;
    const n = input.length;
    const out = new Float32Array(Math.ceil((n - pos) / step) + 1);
    let k = 0;
    while (pos < n) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const a = i === 0 ? last : input[i - 1];
      const b = input[i];
      out[k++] = a + (b - a) * frac;
      pos += step;
    }
    pos -= n;
    last = input[n - 1];
    return out.subarray(0, k);
  }

  function reset() {
    last = 0;
    pos = 0;
  }

  return { process, reset };
}

// ---------- TTS 播放 ----------
// 每包解码后按 nextStartTime 首尾相接排队播放。
// 队列播空（首包或网络断流）时，重新留 START_BUFFER 的缓冲再开始，并做短淡入避免爆音。

const START_BUFFER_S = 0.5; // 实测句间 TTS 生成空档约 0.6s，需要约 0.25s 缓冲才不断流
const MIN_LEAD_S = 0.02;
const FADE_IN_S = 0.004;

let audioCtx = null;
let decoder = null;
let resampler = null;
let nextStartTime = 0;
const activeSources = new Set();
let turnStats = null;

function newTurnStats() {
  return { frames: 0, seconds: 0, underruns: 0, dropped: 0 };
}

// 必须在用户点击/按键事件里同步调用（iOS 要求）
function unlockAudio() {
  try {
    // 让 iPhone 静音键不影响播放；麦克风启用后要保持 play-and-record，不能改回来
    if (navigator.audioSession && micState === 'off' && navigator.audioSession.type !== 'playback') {
      navigator.audioSession.type = 'playback';
      console.log('[audio] audioSession.type=playback');
    }
  } catch (e) {
    console.log('[audio] audioSession not settable', e);
  }

  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
    audioCtx.onstatechange = () => console.log('[audio] context state=' + audioCtx.state);
    resampler = createResampler(OPUS_RATE, audioCtx.sampleRate);
    console.log('[audio] context created, sampleRate=' + audioCtx.sampleRate);

    // 播一个极短的静音 buffer，兼容较老的 iOS 解锁方式
    const silent = audioCtx.createBuffer(1, 1, audioCtx.sampleRate);
    const src = audioCtx.createBufferSource();
    src.buffer = silent;
    src.connect(audioCtx.destination);
    src.start(0);
  }
  if (audioCtx.state !== 'running') {
    audioCtx.resume().then(
      () => console.log('[audio] context resumed, state=' + audioCtx.state),
      (e) => console.log('[audio] context resume failed', e)
    );
  }
  if (!decoder) decoder = createOpusDecoder();
  TTSPlayer.unlock();
}

function playOpusFrame(buffer) {
  if (!turnStats) turnStats = newTurnStats();
  if (!audioCtx || !decoder) {
    turnStats.dropped++;
    if (turnStats.dropped === 1) console.log('[audio] frame dropped, audio not unlocked yet (tap send first)');
    return;
  }

  const pcm16k = decoder.decode(new Uint8Array(buffer));
  if (!pcm16k) {
    turnStats.dropped++;
    return;
  }

  const now = audioCtx.currentTime;
  const startingRun = nextStartTime < now + MIN_LEAD_S;
  if (startingRun) {
    if (turnStats.frames > 0) {
      turnStats.underruns++;
      console.log('[audio] underrun, late by ' + (now - nextStartTime).toFixed(3) + 's, rebuffering');
    } else {
      console.log('[audio] first frame, start in ' + START_BUFFER_S + 's, ctx state=' + audioCtx.state);
    }
    nextStartTime = now + START_BUFFER_S;
  }

  const pcm = resampler.process(pcm16k);
  if (pcm.length === 0) return;

  const audioBuffer = audioCtx.createBuffer(1, pcm.length, audioCtx.sampleRate);
  const channel = audioBuffer.getChannelData(0);
  channel.set(pcm);
  if (startingRun) {
    const fade = Math.min(channel.length, Math.round(FADE_IN_S * audioCtx.sampleRate));
    for (let i = 0; i < fade; i++) channel[i] *= i / fade;
  }

  const src = audioCtx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(audioCtx.destination);
  src.onended = () => activeSources.delete(src);
  src.start(nextStartTime);
  activeSources.add(src);

  const duration = pcm.length / audioCtx.sampleRate;
  nextStartTime += duration;
  turnStats.frames++;
  turnStats.seconds += duration;
}

// tts stop：server 已发完本轮音频。重置解码/重采样状态和统计，
// 已排队的音频继续播完（立即停掉会截掉句尾）。
function resetPlaybackQueue() {
  const s = turnStats || newTurnStats();
  const tail = audioCtx ? Math.max(0, nextStartTime - audioCtx.currentTime) : 0;
  console.log('[audio] turn done: frames=' + s.frames + ' audio=' + s.seconds.toFixed(2) + 's underruns=' +
    s.underruns + ' dropped=' + s.dropped + ' remaining=' + tail.toFixed(2) + 's');
  turnStats = null;
  if (decoder) decoder.reset();
  if (resampler) resampler.reset();
}

// 本轮 TTS 是否还在进行：server 未发 tts stop，或本地还有没播完的音频
function isBotSpeaking() {
  const queued = audioCtx ? nextStartTime - audioCtx.currentTime > 0.05 : false;
  return serverSpeaking || queued || TTSPlayer.isSpeaking();
}

// 断线或打断时立即停止所有已排队的音频
function stopAllAudio() {
  for (const src of activeSources) {
    try {
      src.stop();
    } catch (e) {
      // 已经结束的 source 调 stop 会抛错，忽略
    }
  }
  activeSources.clear();
  TTSPlayer.stop();
  nextStartTime = 0;
  turnStats = null;
  if (decoder) decoder.reset();
  if (resampler) resampler.reset();
  console.log('[audio] all playback stopped');
}

// ---------- WebSocket ----------

const HELLO_TIMEOUT_MS = 5000;

let ws = null;
let sessionId = null;
let helloTimer = null;
let serverSpeaking = false; // tts start → true，tts stop → false
// 发 abort 后，server 发送队列里可能还有 0~1 帧旧音频在路上；丢弃直到下一轮 tts start
let dropStaleAudio = false;

function connect() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;

  const url = buildWsUrl();
  console.log('[ws] connecting', url);
  setStatus('connecting', 'Connecting…');
  sessionId = null;

  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    console.log('[ws] open');
    setStatus('connecting', 'Handshaking…');
    sendHello();
  };

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      handleText(event.data);
    } else if (dropStaleAudio) {
      console.log('[audio] dropped stale frame after abort, ' + event.data.byteLength + ' bytes');
    } else {
      playOpusFrame(event.data);
    }
  };

  ws.onerror = (event) => {
    console.log('[ws] error', event);
  };

  ws.onclose = (event) => {
    console.log('[ws] close code=' + event.code + ' reason=' + (event.reason || '(none)') + ' clean=' + event.wasClean);
    clearTimeout(helloTimer);
    ws = null;
    sessionId = null;
    serverSpeaking = false;
    dropStaleAudio = false;
    resetTurn();
    hideTyping();
    setTranscribing(false);
    cancelRecording('socket closed');
    stopAllAudio();
    setStatus('disconnected', 'Disconnected');
    addNotice('Connection lost');
  };
}

function sendJson(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.log('[ws] send skipped, socket not open', obj);
    return false;
  }
  const text = JSON.stringify(obj);
  console.log('[ws] send', text);
  ws.send(text);
  return true;
}

function sendHello() {
  sendJson({
    type: 'hello',
    version: 1,
    transport: 'websocket',
    // server_tts:false —— server 只下发回复文本，不合成语音（也就不会把文本发给 Edge TTS）
    features: { mcp: false, server_tts: false },
    audio_params: { format: 'opus', sample_rate: 16000, channels: 1, frame_duration: 60 },
  });
  clearTimeout(helloTimer);
  helloTimer = setTimeout(() => {
    console.log('[ws] hello timeout, no reply in ' + HELLO_TIMEOUT_MS + 'ms');
    setStatus('connecting', 'Handshake timed out');
  }, HELLO_TIMEOUT_MS);
}

function handleText(raw) {
  let msg;
  try {
    msg = JSON.parse(raw);
  } catch (e) {
    console.log('[ws] non-JSON text', raw);
    return;
  }
  console.log('[ws] recv', msg.type, msg);

  switch (msg.type) {
    case 'hello':
      clearTimeout(helloTimer);
      sessionId = msg.session_id;
      console.log('[ws] hello ok, session_id=' + sessionId + ' audio_params=' + JSON.stringify(msg.audio_params));
      setStatus('ready', 'Connected');
      requestStatus();
      break;

    case 'stt':
      // 就绪通知 {state:"listening"} 没有 text
      if (msg.text) {
        TTSPlayer.stop();
        resetTurn();
        setTranscribing(false);
        addBubble('user', msg.text);
        showTyping();
      }
      break;

    case 'coach':
      // get_diabetes_data 的进度和数据依据（server/plugins_func/functions/get_diabetes_data.py）
      if (msg.state === 'step') {
        showTyping(STEP_LABELS[msg.step]);
      } else if (msg.state === 'evidence') {
        pendingEvidence = msg;
        attachEvidence();
        requestStatus(); // 分析前刚同步过 Nightscout
      }
      break;

    case 'status':
      onStatus(msg);
      break;

    case 'tts':
      if (msg.state === 'start') {
        serverSpeaking = true;
        dropStaleAudio = false;
      } else if (msg.state === 'sentence_start' && msg.text) {
        onAssistantText(msg.text);
      } else if (msg.state === 'stop') {
        console.log('[ws] tts stop, turn finished');
        serverSpeaking = false;
        hideTyping();
        resetPlaybackQueue();
        // 所有 sentence_start（含全文）都先于 stop 到达，此时 botText 是完整回复
        if (botText) TTSPlayer.speak(botText);
      }
      break;

    case 'llm':
      console.log('[ws] emotion', msg.emotion, msg.text);
      break;

    default:
      console.log('[ws] unhandled type', msg.type);
  }
}

// ---------- 麦克风 / 按住说话 ----------
// 首次按下时申请麦克风并加载 mic-worklet.js；授权完成时如果还按着，直接开始录音。
// 按下：（TTS 播放中则先清空播放队列并发 abort）→ listen start → 逐帧 Opus 编码上传
// 松开：worklet 补齐最后一帧 → 等 STOP_DELAY_MS（让 server 把音频帧搬进 ASR 缓冲）→ listen stop

const STOP_DELAY_MS = 150;
const STOP_FALLBACK_MS = 600; // worklet 没回 stopped 时的兜底
const MIN_FRAMES = 5;         // 少于 0.3s 基本识别不出来，给提示

const $talk = document.getElementById('talkBtn');

let micState = 'off'; // 'off' | 'initializing' | 'ready'
let micStream = null;
let micSource = null;
let micNode = null;
let encoder = null;
let pressing = false;      // 手指/鼠标是否按着
let recording = false;     // 已发 listen start，正在上传
let awaitingStop = false;  // 已松开，等 worklet 发完最后一帧
let stopFallbackTimer = null;
let recStats = null;

const $talkLabel = document.getElementById('talkLabel');

function updateTalkButton() {
  const ready = !!sessionId;
  $talk.disabled = !ready;
  $talk.classList.toggle('recording', recording);
  if (micState === 'initializing') $talkLabel.textContent = 'Enabling mic…';
  else if (recording) $talkLabel.textContent = 'Release to send';
  else $talkLabel.textContent = 'Hold to talk';
  if (!recording) $talk.style.setProperty('--level', '0');
  updatePhase();
}

// 按钮上方的一行状态：录音 / 识别 / 思考 / 播报 / 空闲提示
function updatePhase() {
  let text = '';
  let cls = '';
  const speaking = !!sessionId && !recording && isBotSpeaking();
  if (!sessionId) text = '';
  else if (recording) { text = 'Listening…'; cls = 'recording'; }
  else if (awaitingStop || transcribing) { text = 'Got it, transcribing…'; cls = 'active'; }
  else if (waitingReply) { text = 'Joe is thinking…'; cls = 'active'; }
  else if (speaking) { text = 'Joe is speaking · hold to interrupt'; cls = 'active'; }
  else text = 'Hold the button and speak, or type below';
  if ($phase.textContent !== text) $phase.textContent = text;
  $phase.className = cls;
  $talk.classList.toggle('speaking', speaking);
}

async function initMic() {
  micState = 'initializing';
  updateTalkButton();
  try {
    unlockAudio();
    try {
      if (navigator.audioSession) {
        navigator.audioSession.type = 'play-and-record';
        console.log('[mic] audioSession.type=play-and-record');
      }
    } catch (e) {
      console.log('[mic] audioSession not settable', e);
    }

    // 关掉回声消除/降噪/自动增益：按住说话是半双工，用不到；
    // 而且 iOS 开启语音处理后 TTS 播放音量会明显变小
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const track = micStream.getAudioTracks()[0];
    console.log('[mic] permission granted, track settings=' + JSON.stringify(track.getSettings()));
    track.onended = () => {
      console.log('[mic] track ended (interrupted or revoked), will re-init on next press');
      cancelRecording('track ended');
      teardownMic();
    };

    await audioCtx.audioWorklet.addModule('mic-worklet.js');
    micSource = audioCtx.createMediaStreamSource(micStream);
    micNode = new AudioWorkletNode(audioCtx, 'mic-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    micNode.port.onmessage = onMicMessage;
    // 节点要连到 destination 才会被驱动；输出是静音
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    micSource.connect(micNode);
    micNode.connect(mute);
    mute.connect(audioCtx.destination);

    if (!encoder) encoder = createOpusEncoder();
    if (!encoder) throw new Error('opus encoder unavailable');

    micState = 'ready';
    console.log('[mic] ready, context sampleRate=' + audioCtx.sampleRate + ' state=' + audioCtx.state);
    if (pressing) startRecording();
  } catch (e) {
    console.log('[mic] init failed', e);
    addNotice('Microphone unavailable: ' + (e && e.message ? e.message : e));
    teardownMic();
  }
  updateTalkButton();
}

function teardownMic() {
  try {
    if (micSource) micSource.disconnect();
    if (micNode) micNode.disconnect();
  } catch (e) {
    // 已断开，忽略
  }
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  micStream = null;
  micSource = null;
  micNode = null;
  micState = 'off';
}

function startRecording() {
  if (!sessionId || recording || awaitingStop || micState !== 'ready') return;

  if (isBotSpeaking()) {
    console.log('[mic] barge-in while TTS playing: clear playback + abort');
    stopAllAudio();
    serverSpeaking = false;
    dropStaleAudio = true;
    sendJson({ type: 'abort' });
  }

  encoder.reset();
  recStats = { frames: 0, bytes: 0, peak: 0, t0: performance.now() };
  recording = true;
  sendJson({ type: 'listen', mode: 'manual', state: 'start' });
  micNode.port.postMessage({ cmd: 'start' });
  console.log('[mic] recording started');
  updateTalkButton();
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  awaitingStop = true;
  micNode.port.postMessage({ cmd: 'stop' });
  clearTimeout(stopFallbackTimer);
  stopFallbackTimer = setTimeout(() => {
    console.log('[mic] worklet did not confirm stop in ' + STOP_FALLBACK_MS + 'ms, sending listen stop anyway');
    finishRecording();
  }, STOP_FALLBACK_MS);
  updateTalkButton();
}

function finishRecording() {
  if (!awaitingStop) return;
  awaitingStop = false;
  clearTimeout(stopFallbackTimer);
  const s = recStats;
  setTimeout(() => {
    sendJson({ type: 'listen', mode: 'manual', state: 'stop' });
    console.log('[mic] recording done: frames=' + s.frames + ' audio=' + (s.frames * 0.06).toFixed(2) + 's bytes=' + s.bytes +
      ' peak=' + s.peak.toFixed(3) + ' held=' + ((performance.now() - s.t0) / 1000).toFixed(2) + 's');
    if (s.frames < MIN_FRAMES) addNotice('Too short — hold the button while you speak');
    else if (s.peak < 0.01) addNotice('No sound picked up — check the microphone');
    else setTranscribing(true);
    updatePhase();
  }, STOP_DELAY_MS);
}

// 断线/中断时丢弃本次录音，不发 listen stop
function cancelRecording(reason) {
  if (!recording && !awaitingStop) return;
  console.log('[mic] recording cancelled: ' + reason);
  recording = false;
  awaitingStop = false;
  clearTimeout(stopFallbackTimer);
  if (micNode) micNode.port.postMessage({ cmd: 'stop' });
  updateTalkButton();
}

function onMicMessage(event) {
  const msg = event.data;
  if (msg.type === 'ready') {
    console.log('[mic] worklet ready, input sampleRate=' + msg.sampleRate + ' → 16000');
  } else if (msg.type === 'frame') {
    if (!recording && !awaitingStop) return;
    const packet = encoder.encode(msg.pcm);
    if (!packet || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(packet);
    recStats.frames++;
    recStats.bytes += packet.length;
    if (msg.peak > recStats.peak) recStats.peak = msg.peak;
    if (recStats.frames === 1) console.log('[mic] first frame sent, ' + packet.length + ' bytes');
    if (recording) $talk.style.setProperty('--level', Math.min(1, msg.peak * 3).toFixed(2));
  } else if (msg.type === 'stopped') {
    finishRecording();
  }
}

function pressStart(source) {
  if (pressing || $talk.disabled) return;
  pressing = true;
  console.log('[mic] press (' + source + ')');
  unlockAudio();
  if (micState === 'ready') startRecording();
  else if (micState === 'off') initMic();
  updateTalkButton();
}

function pressEnd(source) {
  if (!pressing) return;
  pressing = false;
  console.log('[mic] release (' + source + ')');
  // touchend/mouseup 是 iOS 认可的用户手势，再解锁一次音频
  unlockAudio();
  stopRecording();
  updateTalkButton();
}

// touch：preventDefault 阻止随后合成的 mouse 事件、长按菜单和页面滚动
$talk.addEventListener('touchstart', (e) => { e.preventDefault(); pressStart('touch'); }, { passive: false });
$talk.addEventListener('touchend', (e) => { e.preventDefault(); pressEnd('touch'); }, { passive: false });
$talk.addEventListener('touchcancel', (e) => { e.preventDefault(); pressEnd('touchcancel'); }, { passive: false });
// mouse：在按钮上按下，在窗口任意位置松开都算
$talk.addEventListener('mousedown', (e) => { if (e.button === 0) pressStart('mouse'); });
window.addEventListener('mouseup', (e) => { if (e.button === 0) pressEnd('mouse'); });
$talk.addEventListener('contextmenu', (e) => e.preventDefault());
// 切到后台时当作松开
document.addEventListener('visibilitychange', () => { if (document.hidden) pressEnd('hidden'); });

// ---------- 输入 ----------

// 文字输入、开场建议和 Today 面板的快捷提问都走这里；用户气泡等 server 回 stt 再显示
function sendText(text) {
  if (!text || !sessionId) return false;
  return sendJson({ type: 'listen', mode: 'manual', state: 'detect', text: text });
}

$form.addEventListener('submit', (e) => {
  e.preventDefault();
  // 发送按钮/键盘回车是用户手势，在这里解锁音频（iOS 要求）
  unlockAudio();
  if (sendText($input.value.trim())) $input.value = '';
});

$reconnect.addEventListener('click', () => {
  console.log('[ws] manual reconnect');
  unlockAudio();
  connect();
});

// 首次点击页面任意位置也解锁 TTS（iOS 要求首次发声在用户手势里）
document.addEventListener('touchend', () => TTSPlayer.unlock(), { once: true });
document.addEventListener('click', () => TTSPlayer.unlock(), { once: true });

// 状态行跟随 TTS 播放（本地播放结束没有事件，轮询）；胶囊上的「几分钟前」每 30s 刷新
setInterval(updatePhase, 300);
setInterval(renderPill, 30000);
setInterval(() => { if (!document.hidden) requestStatus(); }, STATUS_POLL_MS);
document.addEventListener('visibilitychange', () => { if (!document.hidden) requestStatus(); });
$starters.appendChild(askButtons('asks'));

TTSPlayer.init();
console.log('[ws] device-id=' + deviceId + ' client-id=' + clientId + (clientId === JOE_CLIENT_ID ? ' (Joe)' : ' (override)'));
connect();
