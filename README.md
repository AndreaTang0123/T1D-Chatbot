# T1D-Chatbot

面向一型糖尿病（T1D）的语音陪伴助手 "Joe"。用户在 iPhone Safari 或桌面浏览器上按住说话或打字，server 识别语音，结合 Nightscout 的 CGM（血糖）和胰岛素泵数据生成回复，再由浏览器朗读出来。

server 基于开源项目 [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server)（MIT，见 `server/LICENSE`）改写，已经去掉 ESP32 硬件相关的部分，只服务浏览器客户端。

## 目录

| 目录 | 内容 |
|---|---|
| `server/` | Python WebSocket server（端口 8000，路径 `/xiaozhi/v1/`）：ASR → 意图分类 → 注入 CGM/泵/搜索/历史上下文 → LLM 三层 prompt（decision → interpretation → persona） |
| `client/` | 浏览器客户端：纯 HTML + 原生 JS，加一个 `serve.py`（静态页 + Gemini TTS 接口，端口 5173）。详见 [client/README.md](client/README.md)，协议见 [client/PROTOCOL.md](client/PROTOCOL.md) |
| `analysis/` | 离线 CGM 统计脚本（日、周、月的 TIR / CV / GMI、高低血糖事件），研究用，不依赖 server |

T1D 相关的核心代码：

| 文件 | 作用 |
|---|---|
| `server/core/connection.py` | 每个连接的主流程：意图路由、上下文注入、三层 prompt、会话记忆 |
| `server/core/utils/cgm_intent.py` | 意图分类器：判断要不要 CGM、泵、网络搜索，以及时间、天气、地点、退出的快速回答 |
| `server/core/utils/cgm_manager.py` | 每 15 分钟从 Nightscout 拉取血糖，计算 TIR/TAR/TBR/GMI |
| `server/core/utils/pump_manager.py` | 拉取泵事件和 profile，分析 bolus 效果与异常 |
| `server/core/utils/prompt_manager.py` | 加载三层 prompt，支持按 client-id 覆盖 |
| `server/core/utils/history_rag.py`、`web_search.py` | 历史对话检索、DuckDuckGo 搜索 |
| `server/data/<client-id>/prompts/` | Joe 的三层 prompt：`prompt_persona.txt`、`prompt_decision.txt`、`prompt_analysis.txt` |

## 运行

需要：macOS 或 Linux，Python 3.11，`ffmpeg`，以及 [Ollama](https://ollama.com)。

### 1. Ollama

```bash
ollama serve                 # 保持运行
ollama pull qwen2.5:3b       # 意图分类用（本地）
```

主 LLM 在 `server/config.yaml` 的 `LLM.openai.model_name`，目前是 `gemma4:31b-cloud`（Ollama **云端**模型）。

### 2. server

```bash
cd server
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
.venv311/bin/python app.py   # 约 15 秒后监听 8000；第一次启动会从 modelscope 下载 ASR 模型
```

### 3. Joe 的 Nightscout 配置

Joe 的 client-id 是 `26ea0ba9-2d55-4368-a56d-19c4a27c0772`（`client/app.js` 里的 `JOE_CLIENT_ID`）。

```bash
cd server/data/26ea0ba9-2d55-4368-a56d-19c4a27c0772
cp config.example.json config.json   # 填写 Nightscout 地址和 API secret 的 SHA-1
```

没有 `config.json` 时 Joe 照样能对话，只是没有 CGM 和泵数据。

### 4. 浏览器客户端

```bash
cd client
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                 # 可选：填 GEMINI_API_KEY 启用云端语音
.venv/bin/python serve.py            # 127.0.0.1:5173
```

本机调试：打开 `http://127.0.0.1:5173/?ws=ws://127.0.0.1:8000/xiaozhi/v1/`。

### 5. 手机访问（Tailscale HTTPS）

iOS 要求 HTTPS 才能使用麦克风。页面和 WebSocket 走同一个域名：

```bash
tailscale serve --bg --set-path / http://127.0.0.1:5173
tailscale serve --bg --set-path /xiaozhi/v1 http://127.0.0.1:8000/xiaozhi/v1
```

### 6. 离线分析

```bash
cd analysis
export NIGHTSCOUT_URL=https://your-nightscout.example.com
export NIGHTSCOUT_API_SECRET_SHA1=<sha1>
python -m cgm_analyze.daily_summary   # 日期写在各脚本的 main() 里
```

需要 `requests`、`pytz`、`pandas`、`numpy`。

## 数据与隐私

**`server/data/` 不进 git**（见 `.gitignore`）。这个目录里有血糖和泵的 CSV、对话记忆、录音，以及带 Nightscout 密钥的 `config.json`。只有 server 启动必需的 `.config.yaml`、`agent-base-prompt.txt`，以及 Joe 的 prompt 和 `config.example.json` 会提交。

运行时会离开本机的数据：

| 环节 | 去向 |
|---|---|
| 语音识别 | 本地 FunASR，不离开本机 |
| 意图分类 | 本地 `qwen2.5:3b`，不离开本机 |
| 回复生成 | Ollama 云端 `gemma4:31b-cloud`：persona、CGM/泵数据、对话内容都会发出去。改成本地模型才能全本地 |
| 朗读 | `cloud_first` 发给 Google Gemini TTS；`local_only` 用浏览器自带语音 |
| 网络搜索 | 意图分类判断需要时，查询词发给 DuckDuckGo |
