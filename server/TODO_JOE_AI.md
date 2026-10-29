# Joe AI - Future Features TODO

## Ready for Testing
- [x] Multilingual support (responds in user's language)
- [x] CGM status always in context
- [x] CGM trends injection when asked
- [x] News RAG injection when asked
- [x] History RAG (past conversations)
- [x] Concise, casual persona

## Planned Features
- [ ] Schedule/Calendar awareness
  - Integrate with Google Calendar or iCloud
  - Detect queries like "what do I have tomorrow?"
  - Add `needs_schedule` to classifier
  
- [ ] Class notes RAG
  - Index notes/documents for specific classes
  - Useful for study help, assignment reminders
  - Add `needs_class_notes` to classifier

- [ ] Activity tracking (LOCAL on device, exposed via MCP)
  - Use gyroscope + accelerometer for step counting
  - Detect activity level (walking, stationary, sleeping)
  - Expose as MCP tool: `get_activity_stats`
  - Similar to existing `get_battery_level` pattern
  - Server can query when relevant to conversation


## Architecture Notes
- Fast answers (time / weather / location, rule-based): `core/utils/fast_intent.py`
- Context injection (history RAG + current CGM reading): `core/connection.py` `chat()`
- Tools the main LLM can call: `config.yaml` `Intent.function_call.functions`
  - `get_diabetes_data` -> `core/utils/coach_pipeline.py` (analysis -> decision -> interpretation)
- Client configs: `data/<client-id>/config.json`
- Persona prompts: `data/<client-id>/prompts/prompt_persona.txt` (or `prompt.txt`)

To add a new context type:
1. Register a tool in `plugins_func/functions/` (`@register_function`)
2. Add it to `Intent.function_call.functions` in `config.yaml`
3. (Optional) Add config flag in client's `config.json`
