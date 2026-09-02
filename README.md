# T1D-Chatbot

A Type-1-Diabetes voice-coaching layer built on top of the
[xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) Python server.

The chatbot listens on the server's existing WebSocket channel (ESP32 device →
ASR → LLM → TTS), and adds:

* **Nightscout CGM + insulin-pump ingestion** as background tasks
* **An LLM intent classifier** that decides which context (CGM / pump / news /
  web search) a question actually needs
* **A three-layer coaching pipeline** — decision → interpretation → persona —
  so glucose data is prioritised, translated into plain language, and finally
  spoken in a warm, short, friend-like voice
* **Offline CGM analytics scripts** (daily / weekly / monthly summaries) for
  research and prompt development

> **This repository is not a standalone server.** It contains the modified and
> newly-added modules only. They must be copied into a checkout of
> `xiaozhi-esp32-server` (see [Installation](#installation)) — every file here
> imports from `config.*`, `core.*`, and `plugins_func.*`, which live in the
> upstream project.

---

## Where is the WebSocket endpoint?

**It is not defined in this repository.** No IP, port, or path is configured in
any file here.

[connection.py](connection.py) is the upstream `core/connection.py`. It only
*handles* a socket that has already been accepted:

| What | Where |
| --- | --- |
| `ConnectionHandler.handle_connection(ws)` | [connection.py:185](connection.py#L185) |
| Reads `device-id` / `client-id` request headers | [connection.py:191-205](connection.py#L191-L205) |
| Reads the request path (`?from=mqtt_gateway`) | [connection.py:220](connection.py#L220) |
| Message read loop | [connection.py:239](connection.py#L239) |

The listener itself (bind address, port, URL path) lives in the upstream
project, not here:

* `main/xiaozhi-server/core/websocket_server.py` — creates the `websockets`
  server and registers the route
* `main/xiaozhi-server/data/.config.yaml` (or `config.yaml`) — `server.ip` and
  `server.port`

In a default xiaozhi-esp32-server install the device connects to
`ws://<server-ip>:8000/xiaozhi/v1/` (bind `0.0.0.0`, port `8000`). Confirm the
values for your deployment with:

```bash
grep -nA5 "^server:" main/xiaozhi-server/data/.config.yaml
grep -rn "websockets.serve\|/xiaozhi/v1" main/xiaozhi-server/core/websocket_server.py
```

The only network endpoint hard-coded in this repo is the **Nightscout REST API**
in [fetch/fetch_cgm.py:8](fetch/fetch_cgm.py#L8) — see
[Security notes](#security-notes).

---

## Repository layout

| File | Role | Upstream destination |
| --- | --- | --- |
| [connection.py](connection.py) | Per-connection handler: intent routing, RAG/CGM/pump context injection, coaching pipeline, TTS dispatch, session memory | `core/connection.py` |
| [cgm_intent.py](cgm_intent.py) | LLM intent classifier + fast answers (time / weather / location / battery / volume / exit) | `core/utils/cgm_intent.py` |
| [cgm_manager.py](cgm_manager.py) | Nightscout `entries.json` polling, CSV store, TIR/TAR/TBR/GMI metrics, hourly pattern detection | `core/utils/cgm_manager.py` |
| [pump_manager.py](pump_manager.py) | Nightscout `treatments.json` + `profile.json` polling, bolus/temp-basal normalisation, insulin-effectiveness and anomaly analysis | `core/utils/pump_manager.py` |
| [prompt_manager.py](prompt_manager.py) | Loads the layered prompts (persona / decision / interpretation) with per-client overrides; renders the Jinja base template with time, location, weather | `core/utils/prompt_manager.py` |
| [prompt_persona.txt](prompt_persona.txt) | Layer 3 — "Joe", the short, warm, language-matching voice | `data/prompts/` or `data/<client_id>/` |
| [prompt_decision.txt](prompt_decision.txt) | Layer 1 — safety-first prioritisation of CGM/pump signals | same |
| [prompt_analysis.txt](prompt_analysis.txt) | Layer 2 — turns the decision output into one plain-language insight | same |
| [prompt.txt](prompt.txt) | Legacy single-prompt slot (currently empty) | `data/<client_id>/prompt.txt` |
| [fetch/fetch_cgm.py](fetch/fetch_cgm.py) | Standalone paged Nightscout CGM fetcher used by the analysis scripts | research-only |
| [fetch/fetch_pump.py](fetch/fetch_pump.py) | Earlier standalone copy of the pump fetcher (superseded by `pump_manager.py`) | research-only |
| [cgm_analyze/daily_summary.py](cgm_analyze/daily_summary.py) | One local day: TIR/TBR/TAR, SD, CV, GMI, hypo/hyper event detection | research-only |
| [cgm_analyze/weekly_summary.py](cgm_analyze/weekly_summary.py) | 7-day rollup | research-only |
| [cgm_analyze/monthly_summary.py](cgm_analyze/monthly_summary.py) | 30-day rollup | research-only |

---

## How a turn is processed

```
device --ws--> ConnectionHandler.handle_connection()   connection.py:185
                      |
                      v
             ConnectionHandler.chat(query)             connection.py:1155
                      |
        1. classify_context_needs()                    cgm_intent.py:283
           -> {fast_answer, needs_cgm, needs_pump,
               needs_news, needs_search, language, reply}
                      |
        2. fast answer? (time/weather/location/        connection.py:1195
           battery/volume/exit) -> straight to TTS, main LLM skipped
                      |
        3. context injection                           connection.py:1258+
           - history_rag.search()
           - news_rag.search()          (if needs_news and enabled)
           - CGMManager.get_context_summary()   (if data/<id>/config.json has "cgm")
           - PumpManager.get_context_summary()  (if it has "pump")
                      |
        4. run_coach_pipeline()                        connection.py:1123
           decision prompt  -> JSON of prioritised issues
           analysis prompt  -> 1-2 sentence plain-language insight
           persona prompt   -> final short spoken reply
                      |
                      v
                 TTS queue -> device
```

Steps 1 and 4 each use `_llm_single_shot()`
([connection.py:978](connection.py#L978)); the intent classifier caches its JSON
result per `(client_id, query, language_hint)`.

### The three prompt layers

1. **Decision** ([prompt_decision.txt](prompt_decision.txt)) — receives the
   structured analysis built by `_build_pipeline_analysis_result()`
   ([connection.py:1028](connection.py#L1028)) and picks what matters, ordered
   safety → control → patterns → trends. It never explains.
2. **Interpretation** ([prompt_analysis.txt](prompt_analysis.txt)) — turns the
   selected issues into one or two sentences of pattern language ("you tend to
   spike after dinner"), never raw metrics.
3. **Persona** ([prompt_persona.txt](prompt_persona.txt)) — "Joe": one short
   sentence, plain text, the user's exact language, and glucose is mentioned
   unprompted only below 70 or above 250 mg/dL.

Per-client overrides win over the project-level files: `PromptManager` first
looks for `data/<client_id>/<layer>.txt`, then `data/<device_id>/<layer>.txt`,
then the configured path ([prompt_manager.py:130-155](prompt_manager.py#L130-L155)).

---

## Per-client data layout

Everything is keyed by `client_id` (the `client-id` WebSocket header, falling
back to `device-id`):

```
data/
└── <client_id>/
    ├── config.json          # secrets + feature flags (below)
    ├── cgm.csv              # time, sgv, direction, unix_s, hour, weekday
    ├── pump_events.csv      # normalised bolus / temp-basal events
    ├── pump_profile.json    # active Nightscout profile
    ├── prompt.txt           # optional legacy single-prompt override
    ├── prompt_persona.txt   # optional per-client layer overrides
    ├── prompt_decision.txt
    └── prompt_analysis.txt
```

`config.json`:

```json
{
  "location": "Durham",
  "news_rag_enabled": false,
  "cgm": {
    "base_url": "https://your-nightscout.example.com",
    "api_secret": "<sha1 of your Nightscout API secret>",
    "user_tz": "US/Eastern"
  },
  "pump": {
    "base_url": "https://your-nightscout.example.com",
    "api_secret": "<sha1 of your Nightscout API secret>",
    "user_tz": "US/Eastern"
  }
}
```

A client without a `cgm` / `pump` block is simply skipped by the corresponding
background task — the chatbot still works as a plain voice assistant.

### Background ingestion

Both managers expose a task that walks every directory under `data/` every
**15 minutes** (60 s backoff on error) and appends only rows newer than the last
stored `unix_s`:

* `cgm_background_task()` — [cgm_manager.py:345](cgm_manager.py#L345)
* `pump_background_task(data_root="data")` — [pump_manager.py:683](pump_manager.py#L683),
  or `create_pump_background_task()` for use with `asyncio.create_task()`

Schedule them from the server's async startup:

```python
asyncio.create_task(cgm_background_task())
asyncio.create_task(create_pump_background_task("data"))
```

### Derived signals fed to the decision layer

* CGM — latest reading + minutes since, TIR / TAR / TBR over 14 days, mean
  glucose, GMI (`3.31 + 0.02392 × mean`), hours of day that trend high
  ([cgm_manager.py:253-341](cgm_manager.py#L253-L341))
* Pump — latest event and latest bolus, most frequent recent event type,
  typical bolus hour and size, glucose response after recent boluses, and
  anomalies such as bolus stacking or repeated overnight temp basals
  ([pump_manager.py:540-682](pump_manager.py#L540-L682))

---

## Offline analysis scripts

Independent of the server; they hit Nightscout directly and print a report.

```bash
python -m cgm_analyze.daily_summary     # one US/Eastern calendar day
python -m cgm_analyze.weekly_summary    # 7 days
python -m cgm_analyze.monthly_summary   # 30 days
```

The target date is currently a literal inside each `main()` (e.g.
`target_date = "2026-03-07"` in
[cgm_analyze/daily_summary.py:243](cgm_analyze/daily_summary.py#L243)) — edit it
before running. Each reports mean/median/min/max, SD, CV, GMI, TIR 70-180,
TBR <70 and <54, TAR >180 and >250, plus hypo events (≥3 consecutive readings)
and hyper events (≥6 consecutive readings) with start, end, and duration.

---

## Installation

```bash
# 1. Get the upstream server
git clone https://github.com/xinnan-tech/xiaozhi-esp32-server.git
cd xiaozhi-esp32-server/main/xiaozhi-server

# 2. Copy this repo's modules into it
cp <this-repo>/connection.py            core/connection.py
cp <this-repo>/cgm_intent.py            core/utils/cgm_intent.py
cp <this-repo>/cgm_manager.py           core/utils/cgm_manager.py
cp <this-repo>/pump_manager.py          core/utils/pump_manager.py
cp <this-repo>/prompt_manager.py        core/utils/prompt_manager.py
mkdir -p data/prompts
cp <this-repo>/prompt_*.txt             data/prompts/

# 3. Point the config at the layered prompts, then start the server
python app.py
```

Config keys read by `PromptManager`
([prompt_manager.py:51-55](prompt_manager.py#L51-L55)):

```yaml
persona_prompt_template: data/prompts/prompt_persona.txt
decision_prompt_template: data/prompts/prompt_decision.txt
interpretation_prompt_template: data/prompts/prompt_analysis.txt
```

An optional `LLM.fast_intent` block selects a cheaper/faster model for the intent
classifier; without it the classifier falls back to `selected_module.LLM`
([cgm_intent.py:77-93](cgm_intent.py#L77-L93)).

### Requirements

Beyond the upstream server's dependencies: `requests`, `pytz`, `jinja2`,
`websockets`, and — for the analysis scripts only — `pandas` and `numpy`.
Python 3.11 ([pyrightconfig.json](pyrightconfig.json)).

---

## Security notes

* [fetch/fetch_cgm.py:8-9](fetch/fetch_cgm.py#L8-L9) hard-codes a live
  Nightscout URL and a SHA-1 API secret. Move both to environment variables or
  `data/<client_id>/config.json` before this repository is shared or published,
  and rotate the secret.
* [connection.py:206-213](connection.py#L206-L213) force-maps the hard-coded
  device MAC `b0:a6:04:5b:d7:98` to a fixed client UUID. This is a development
  shortcut and should be removed before any multi-user deployment.
* CGM and pump data are written unencrypted to `data/<client_id>/*.csv`.

## Status / known gaps

* `CGMManager.analyze_weekly_trends()` is a stub that returns `[]`
  ([cgm_manager.py:297](cgm_manager.py#L297)); the real weekly/monthly logic
  currently lives only in the offline `cgm_analyze/` scripts.
* `_build_pipeline_analysis_result()` probes for manager methods that do not all
  exist yet (`get_metrics_summary`, `get_latest_status`, `get_recent_patterns`,
  `get_joint_signals`, `detect_anomalies`) via `hasattr`, so those sections are
  silently omitted today ([connection.py:1063-1114](connection.py#L1063-L1114)).
* [fetch/fetch_pump.py](fetch/fetch_pump.py) duplicates most of
  [pump_manager.py](pump_manager.py) at an earlier revision; `pump_manager.py`
  is the one the server uses.
* [prompt.txt](prompt.txt) is empty — the layered prompts have replaced it.
