// TTS 播放组件：云端 Gemini（Puck）优先，失败自动降级到浏览器 speechSynthesis
// 对外：TTSPlayer.init() / unlock() / speak(text) / stop() / isSpeaking()
// 模式由 serve.py 的 TTS_MODE 决定；?tts=local_only 可在前端强制只用浏览器（只能降级，不能升级）

'use strict';

const TTSPlayer = (() => {
  const FIRST_CHUNK_TIMEOUT_MS = 5000;  // 用户在等第一声：超过就改用浏览器
  const NEXT_CHUNK_TIMEOUT_MS = 30000;  // 后续分段在前一段播放时预取，可以宽松些
  const CLOUD_COOLDOWN_MS = 60000;      // 云端失败后这段时间内直接用浏览器，避免每轮都等超时
  // 分段上限逐段放大：实测 Gemini 生成约 2s + 0.019s/字符，音频约 0.06s/字符（生成快约 3 倍），
  // 所以每段都能在上一段播放期间生成好、不断档；一条回复最多 3 次请求、2 个衔接点。
  // 第一段要短，1~2 秒生成完，不会撞上 5 秒超时；最后一段收下剩余全部（serve.py 上限 2000）。
  const CHUNK_LIMITS = [80, 200, 1500];
  const FIRST_CHUNK_CHARS = CHUNK_LIMITS[0];
  const VOICES_WAIT_MS = 1500;          // Safari 有时不触发 voiceschanged

  const LABEL_CLOUD = '✨ Puck Neural Voice';
  const LABEL_BROWSER = '✨ Browser Voice'; // 开始朗读后换成实际的声音名

  const $label = document.getElementById('voiceLabel');
  const audio = new Audio();
  audio.preload = 'auto';

  let mode = 'local_only'; // 拿到 config 之前按最保守处理
  let cloudAvailable = false;
  let cloudFailedAt = 0;
  let cloudBlockedUntil = 0; // 额度用完（429）时按 Google 给的等待时间暂停云端
  let unlocked = false;
  let generation = 0;      // stop() 时自增，让正在进行的 speak 循环退出
  let speaking = false;
  let voices = [];
  const inflight = new Set(); // 进行中的云端请求，stop() 时取消，避免浪费额度

  // ---------- 配置 ----------

  async function init() {
    const forceLocal = new URLSearchParams(location.search).get('tts') === 'local_only';
    try {
      const res = await fetch('/api/tts/config', { cache: 'no-store' });
      const cfg = await res.json();
      mode = forceLocal ? 'local_only' : cfg.mode;
      cloudAvailable = mode === 'cloud_first' && cfg.cloud_available;
      console.log('[tts] config mode=' + mode + ' cloud_available=' + cloudAvailable + (forceLocal ? ' (forced by ?tts=local_only)' : ''));
    } catch (e) {
      mode = 'local_only';
      cloudAvailable = false;
      console.log('[tts] config fetch failed, browser voice only', e);
    }
    setLabel(cloudUsable());
    loadVoices();
  }

  function cloudUsable() {
    const now = Date.now();
    return mode === 'cloud_first' && cloudAvailable && now - cloudFailedAt > CLOUD_COOLDOWN_MS && now > cloudBlockedUntil;
  }

  function setLabel(cloud) {
    if ($label) $label.textContent = cloud ? LABEL_CLOUD : LABEL_BROWSER;
  }

  // ---------- 浏览器声音 ----------
  // getVoices() 异步加载：先读一次，再监听 voiceschanged

  function loadVoices() {
    if (!window.speechSynthesis) {
      console.log('[tts] speechSynthesis not supported');
      return;
    }
    const update = () => {
      voices = speechSynthesis.getVoices();
      console.log('[tts] voices loaded: ' + voices.length + ' en=' + voices
        .filter((v) => v.lang && v.lang.toLowerCase().startsWith('en'))
        .map((v) => v.name + (isQuality(v) ? '*' : '')).join(','));
    };
    update();
    speechSynthesis.addEventListener('voiceschanged', update);
  }

  let voicesWaited = false; // 只等一次：有的浏览器列表一直为空，之后直接用系统默认

  function waitForVoices() {
    if (voices.length || voicesWaited || !window.speechSynthesis) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        speechSynthesis.removeEventListener('voiceschanged', done);
        voices = speechSynthesis.getVoices();
        voicesWaited = true;
        resolve();
      };
      speechSynthesis.addEventListener('voiceschanged', done);
      setTimeout(done, VOICES_WAIT_MS);
    });
  }

  // 名字带 Google 的是联网合成（Chrome），localService=false 的同理，都排到最后
  function isNetworkVoice(v) {
    return /google/i.test(v.name) || v.localService === false;
  }

  // Apple 的特效声音和 Eloquence 复古声音：不适合 Joe，列表和自动选择里都去掉
  const NOVELTY_NAMES = new Set(['albert', 'bad news', 'bahh', 'bells', 'boing', 'bubbles', 'cellos', 'deranged',
    'fred', 'good news', 'hysterical', 'jester', 'junior', 'kathy', 'organ', 'pipe organ', 'princess', 'ralph',
    'superstar', 'trinoids', 'whisper', 'wobble', 'zarvox']);
  const RETRO_NAMES = new Set(['eddy', 'flo', 'grandma', 'grandpa', 'reed', 'rocko', 'sandy', 'shelley']);

  function isNoveltyVoice(v) {
    const base = v.name.toLowerCase().replace(/\s*\(.*\)\s*$/, '').trim(); // 去掉 "(English (US))" 之类后缀
    return NOVELTY_NAMES.has(base) || RETRO_NAMES.has(base) || /eloquence/i.test(v.voiceURI || '');
  }

  function usableVoices(prefix) {
    return voices.filter((v) => v.lang && v.lang.toLowerCase().startsWith(prefix) && !isNetworkVoice(v) && !isNoveltyVoice(v));
  }

  // iOS 上增强/高级声音的 name 不一定带后缀，但 voiceURI 形如 com.apple.voice.enhanced.en-US.Allison
  function isQuality(v) {
    return /premium|enhanced|高级|优化/i.test(v.name + ' ' + (v.voiceURI || ''));
  }

  const PREFERRED = {
    en: [/^Allison\b/i, /^Ava\b.*premium/i, /^Zoe\b.*premium/i, /^Evan\b.*enhanced/i],
    zh: [/(tingting|婷婷).*(premium|enhanced|高级|优化)/i, /(meijia|美佳).*(premium|enhanced|高级|优化)/i],
  };

  // ?voice=名字 可以临时指定声音（只匹配名字开头，不区分大小写）
  const VOICE_OVERRIDE = (new URLSearchParams(location.search).get('voice') || '').trim().toLowerCase();

  // 用户在声音标签的下拉框里选的声音（按语言各存一个 voiceURI），只存在本机浏览器
  const VOICE_KEY = 't1d.voice.';
  const chosenNow = {}; // 本页面内的选择；localStorage 不可用时也能生效

  function savedVoice(prefix) {
    if (prefix in chosenNow) return chosenNow[prefix];
    try {
      return localStorage.getItem(VOICE_KEY + prefix) || '';
    } catch (e) {
      return '';
    }
  }

  function pickVoice(lang) {
    const prefix = lang === 'zh' ? 'zh' : 'en';
    const local = usableVoices(prefix);
    const chosen = savedVoice(prefix);
    if (chosen) {
      const v = local.find((x) => x.voiceURI === chosen);
      if (v) return v;
    }
    if (VOICE_OVERRIDE) {
      const v = local.find((x) => x.name.toLowerCase().startsWith(VOICE_OVERRIDE));
      if (v) return v;
    }
    // 同名声音（如 Allison 普通版和增强版）优先取增强/高级版
    const byQuality = local.filter(isQuality).concat(local.filter((v) => !isQuality(v)));
    for (const re of PREFERRED[lang]) {
      const v = byQuality.find((x) => re.test(x.name));
      if (v) return v;
    }
    const zhCN = (v) => v.lang.toLowerCase().replace('_', '-') === 'zh-cn';
    const ordered = lang === 'zh'
      ? [(v) => isQuality(v) && zhCN(v), isQuality, (v) => /tingting|婷婷/i.test(v.name), zhCN, () => true]
      : [isQuality, (v) => v.default, () => true];
    for (const test of ordered) {
      const v = local.find(test);
      if (v) return v;
    }
    return null; // 没有本地声音：交给系统默认
  }

  // 汉字占比足够高就按中文读
  function detectLang(text) {
    const han = (text.match(/\p{Script=Han}/gu) || []).length;
    const latinWords = (text.match(/[A-Za-z]+/g) || []).length;
    return han > 0 && han >= latinWords ? 'zh' : 'en';
  }

  function speakBrowser(text, gen) {
    if (!window.speechSynthesis) return Promise.resolve();
    return waitForVoices().then(() => new Promise((resolve) => {
      if (gen !== generation) return resolve();
      const lang = detectLang(text);
      const u = new SpeechSynthesisUtterance(text);
      const v = pickVoice(lang);
      if (v) {
        u.voice = v;
        u.lang = v.lang;
      } else {
        u.lang = lang === 'zh' ? 'zh-CN' : 'en-US';
      }
      u.onstart = () => console.log('[tts] browser utterance started, audioSession=' +
        (navigator.audioSession ? navigator.audioSession.type : 'n/a'));
      u.onend = () => {
        console.log('[tts] browser utterance ended');
        resolve();
      };
      u.onerror = (e) => {
        console.log('[tts] browser utterance error', e.error);
        resolve();
      };
      if ($label) $label.textContent = v ? '✨ ' + v.name : LABEL_BROWSER;
      console.log('[tts] browser speak lang=' + lang + ' voice=' + (v ? v.name + ' (' + v.lang + ')' : 'system default') + ' chars=' + text.length);
      speechSynthesis.speak(u);
    }));
  }

  // ---------- 云端 ----------

  async function fetchCloud(text, timeoutMs) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    inflight.add(ctrl);
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        let body = null;
        try { body = await res.json(); } catch (e) { /* 非 JSON */ }
        if (res.status === 429 && body && body.retry_after_s) {
          cloudBlockedUntil = Date.now() + Math.min(body.retry_after_s, 3600) * 1000;
          console.log('[tts] cloud quota exhausted, browser voice for the next ' + Math.min(body.retry_after_s, 3600) + 's');
        }
        const err = new Error('HTTP ' + res.status + ' ' + (body ? JSON.stringify(body) : ''));
        err.status = res.status;
        throw err;
      }
      return URL.createObjectURL(await res.blob());
    } finally {
      clearTimeout(timer);
      inflight.delete(ctrl);
    }
  }

  function failReason(e) {
    return e.name === 'AbortError' ? 'timeout/cancelled' : e.message;
  }

  function playUrl(url, gen) {
    return new Promise((resolve, reject) => {
      if (gen !== generation) return resolve();
      audio.onended = () => resolve();
      audio.onerror = () => reject(new Error('audio element error'));
      audio.src = url;
      audio.play().catch(reject);
    });
  }

  // ---------- 分段 ----------
  // 第一段短一些尽快出声；第一句太长时在逗号处切开。之后的段按 CHUNK_LIMITS 逐段放大。
  // 英文句号/逗号后面要跟空白才算断点，避免把 6.5、1,000 这类数字切开。

  const SENTENCE_RE = /.+?(?:[。！？!?；;\n]+|\.(?=\s|$)|$)/gs;
  const CLAUSE_RE = /.+?(?:[，、：]+|[,:](?=\s|$)|$)/gs;

  function splitChunks(text) {
    let pieces = text.match(SENTENCE_RE) || [text];
    if (pieces[0].length > FIRST_CHUNK_CHARS) {
      pieces = (pieces[0].match(CLAUSE_RE) || [pieces[0]]).concat(pieces.slice(1));
    }
    const chunks = [];
    let cur = '';
    for (const p of pieces) {
      const limit = CHUNK_LIMITS[Math.min(chunks.length, CHUNK_LIMITS.length - 1)];
      if (cur && (cur + p).length > limit) {
        chunks.push(cur);
        cur = '';
      }
      cur += p;
    }
    if (cur.trim()) chunks.push(cur);
    return chunks.map((c) => c.trim()).filter(Boolean);
  }

  // 朗读前去掉 markdown 符号和 emoji
  function cleanForSpeech(text) {
    return text
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/[*_`#>|~]+/g, ' ')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/\p{Extended_Pictographic}/gu, '')
      .replace(/[ \t]+/g, ' ')
      .trim();
  }

  // ---------- 对外接口 ----------

  async function speak(rawText) {
    stop();
    const text = cleanForSpeech(rawText || '');
    if (!text) return;
    const gen = generation;
    speaking = true;
    const chunks = splitChunks(text);
    let useCloud = cloudUsable();
    setLabel(useCloud);
    console.log('[tts] speak ' + text.length + ' chars in ' + chunks.length + ' chunk(s), path=' + (useCloud ? 'cloud' : 'browser'));

    // 后续段失败先重试一次，仍失败才降级（第一段不重试：用户在等第一声）
    const fetchWithRetry = async (chunk) => {
      try {
        return await fetchCloud(chunk, NEXT_CHUNK_TIMEOUT_MS);
      } catch (e) {
        if (gen !== generation || e.status === 429) throw e; // 额度用完重试也没用
        console.log('[tts] cloud chunk failed, retrying once: ' + failReason(e));
        return fetchCloud(chunk, NEXT_CHUNK_TIMEOUT_MS);
      }
    };

    try {
      let pending = useCloud ? fetchCloud(chunks[0], FIRST_CHUNK_TIMEOUT_MS) : null;
      for (let i = 0; i < chunks.length; i++) {
        if (gen !== generation) return;
        if (useCloud) {
          try {
            const t0 = performance.now();
            const url = await pending;
            // 播当前段的同时预取下一段
            pending = i + 1 < chunks.length ? fetchWithRetry(chunks[i + 1]) : null;
            if (pending) pending.catch(() => {}); // 失败在轮到它时处理
            console.log('[tts] cloud chunk ' + (i + 1) + '/' + chunks.length + ' ready in ' + Math.round(performance.now() - t0) + 'ms');
            try {
              await playUrl(url, gen);
            } finally {
              URL.revokeObjectURL(url);
            }
            continue;
          } catch (e) {
            if (gen !== generation) return;
            console.log('[tts] cloud failed, falling back to browser voice: ' + failReason(e));
            cloudFailedAt = Date.now();
            useCloud = false;
            pending = null;
            setLabel(false);
          }
        }
        await speakBrowser(chunks[i], gen);
      }
    } finally {
      if (gen === generation) speaking = false;
    }
  }

  function stop() {
    generation++;
    speaking = false;
    for (const ctrl of inflight) ctrl.abort();
    inflight.clear();
    try {
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
    } catch (e) {
      // 忽略
    }
    if (window.speechSynthesis) speechSynthesis.cancel();
  }

  function isSpeaking() {
    return speaking || (window.speechSynthesis ? speechSynthesis.speaking : false);
  }

  // iOS：<audio> 和 speechSynthesis 都要在用户手势里先发一次声才允许之后自动播放
  const SILENT_WAV = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=';

  function unlock() {
    if (unlocked) return;
    unlocked = true;
    try {
      audio.src = SILENT_WAV;
      audio.play().then(
        () => console.log('[tts] audio element unlocked'),
        (e) => console.log('[tts] audio element unlock failed', e)
      );
    } catch (e) {
      console.log('[tts] audio element unlock error', e);
    }
    if (window.speechSynthesis) {
      const u = new SpeechSynthesisUtterance(' ');
      u.volume = 0;
      speechSynthesis.speak(u);
      console.log('[tts] speechSynthesis unlocked');
    }
  }

  // ---------- 声音选择 ----------
  // 选项在每次点开前重建：iOS 的声音列表可能在页面打开后才加载完

  const $select = document.getElementById('voiceSelect');

  function voiceLabel(v) {
    return v.name + (isQuality(v) ? ' ★' : '') + ' · ' + v.lang;
  }

  function fillPicker() {
    if (!$select || !window.speechSynthesis) return;
    voices = speechSynthesis.getVoices();
    $select.textContent = '';
    for (const prefix of ['en', 'zh']) {
      const list = usableVoices(prefix)
        .sort((a, b) => (isQuality(b) - isQuality(a)) || a.name.localeCompare(b.name));
      const group = document.createElement('optgroup');
      group.label = prefix === 'en' ? 'English' : '中文';
      const auto = document.createElement('option');
      auto.value = prefix + '|';
      auto.textContent = prefix === 'en' ? 'Auto (English)' : '自动（中文）';
      group.appendChild(auto);
      for (const v of list) {
        const o = document.createElement('option');
        o.value = prefix + '|' + v.voiceURI;
        o.textContent = voiceLabel(v);
        group.appendChild(o);
      }
      $select.appendChild(group);
    }
    const current = pickVoice('en');
    $select.value = 'en|' + (savedVoice('en') && current ? current.voiceURI : '');
    console.log('[tts] picker voices en=' + voices.filter((v) => v.lang && v.lang.toLowerCase().startsWith('en'))
      .map((v) => v.name + (isQuality(v) ? '*' : '') + '<' + v.voiceURI + '>').join(','));
  }

  if ($select) {
    $select.addEventListener('pointerdown', fillPicker);
    $select.addEventListener('focus', fillPicker);
    $select.addEventListener('change', () => {
      const [prefix, uri] = $select.value.split('|');
      chosenNow[prefix] = uri;
      try {
        if (uri) localStorage.setItem(VOICE_KEY + prefix, uri);
        else localStorage.removeItem(VOICE_KEY + prefix);
      } catch (e) {
        // 隐私模式等拿不到 localStorage：只在本页面内生效
      }
      const v = pickVoice(prefix);
      console.log('[tts] voice chosen: ' + (v ? v.name + ' <' + v.voiceURI + '>' : 'auto'));
      // 在用户手势里试读一句，顺便解锁 iOS 的朗读
      speak(prefix === 'zh' ? '你好，我是 Joe。' : "Hi, I'm Joe. This is how I'll sound.");
    });
  }

  return { init, unlock, speak, stop, isSpeaking };
})();
