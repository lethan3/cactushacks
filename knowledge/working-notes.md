# Cactushacks — Working Knowledge

## What this is

AI plant-care agent. An LLM (via Ollama) decides when to water, observe, and schedule care for virtual plants. Simulation harness runs the clock, the agent makes decisions each tick.

## Project layout (post-refactor)

```
agent.py               # PlantCareAgent — LLM loop, tool dispatch, context building
harness.py             # CLI simulation runner — clock ticks, plant state updates, agent calls
prompts/
  agent_prompt.txt     # System prompt template (behavioral instructions only)
utils/
  plant.py             # Plant ABC + VirtualPlant (simulation) + ActualPlant (stub)
  clock.py             # SimulatedClock — start_time, current_time, tick()
  priority_queue.py    # TaskQueue — heapq wrapper for scheduled tasks
  logger.py            # AgentLogger — structured JSON log to logs/
  __init__.py          # Exports Plant, VirtualPlant, ActualPlant
requirements.txt       # ollama>=0.4.0
playgrounds/           # Experiments (e2b, smolagents) — not part of main flow
```

## Architecture in one paragraph

`harness.py` creates a `SimulatedClock`, `TaskQueue`, and `VirtualPlant(s)`, then hands them to `PlantCareAgent`. Each simulation step: clock ticks, plant hydration decays, due tasks trigger `agent.act()`. The agent builds a system message (plant info + history + memories), calls `ollama.Client.chat()` with 4 tool definitions, executes any returned tool calls, feeds results back, and loops until the model stops calling tools or `max_rounds` is hit.

## Key design decisions and why

| Decision | Reasoning |
|---|---|
| **ollama SDK with native tool calling** | Replaced raw `requests.post` + regex JSON parsing. SDK auto-generates tool schemas from function signatures, returns structured `tool_calls` objects. Zero parsing code needed. |
| **Tool functions as module-level stubs** | The 4 functions at the top of `agent.py` exist purely for schema generation. The SDK reads their signatures + docstrings to build JSON schemas. Bodies are never called. Actual execution is in `_execute_tool()`. |
| **Multi-round tool calling loop** | `act()` loops up to `max_rounds` times. Each round: model returns tool calls -> agent executes them -> feeds results back as `{"role": "tool"}` messages -> model sees results and decides next action. This lets the model chain: take_picture -> see hydration is low -> water -> add_memory -> schedule next check. |
| **Simulated time, not wall-clock** | `SimulatedClock` decouples from real time. Enables deterministic testing and fast-forward simulation. Plant hydration decays based on hours-passed, not real seconds. |
| **Virtual-first, hardware-later** | `VirtualPlant` fully implemented. `ActualPlant` stubs raise `NotImplementedError`. Same abstract interface — swap at construction time. |
| **Flat tool dispatch (if/elif)** | Considered a registry/decorator pattern but 4 tools don't justify it. The if/elif in `_execute_tool` is readable and grep-able. Revisit if tool count exceeds ~8. |

## Edits made during this session

### 1. Removed unnecessary abstractions (tiers IMMEDIATE through MEDIUM)
- **Deleted entire `tools/` directory** (4 files + `__init__`) — pass-through wrappers over plant methods
- **Stripped `clock.py`** from 44 to 18 lines — removed `get_current_time()`, `get_time_until()`, `get_hours_passed()`, `reset()`; callers use `clock.current_time` directly
- **Stripped `priority_queue.py`** from 88 to 42 lines — removed `is_empty()`, `size()`, `peek_next()`; callers use `len(queue.queue)` and `queue.queue[0]` directly
- **Cleaned `plant.py`** — removed `get_memories()`, `get_recent_history()`, abstract `get_ideal_watering_frequency_days()`; made `ideal_frequency_days` a plain attribute set in `__init__`
- **Cleaned `agent.py`** — removed `set_task_queue()`, `set_clock()`, `_get_tool_schema()`; extracted `_get_plant()` to eliminate duplicated plant-selection logic

### 2. Refactored to ollama SDK
- Replaced `requests.post` to `/api/generate` with `ollama.Client.chat()` using native tool calling
- Deleted `prompts/tool_schema.txt` — schema auto-generated from function signatures
- Deleted 40-line regex JSON parser — SDK returns structured `msg.tool_calls`
- Simplified `agent_prompt.txt` from 38 to 15 lines — no more JSON format instructions
- Added multi-round tool calling loop (model sees tool results, can chain actions)
- Changed `requirements.txt` from `requests` to `ollama`

## Remaining hotspots

### logger.py — untouched, still verbose
- Every `log_X()` method follows the same pattern: build dict, call `_write_entry()`. Could be collapsed to a single `_log(type, **kwargs)` method.
- Still references `reasoning` field in `log_tool_call()` / `log_tool_result()` — this is now always empty string since reasoning comes from the model's conversational output, not embedded in tool calls. Not broken, just vestigial.
- Opens and writes to file on every single log entry (`_write_entry` opens file in append mode each time). For high-frequency logging, should buffer or keep file handle open.

### plant.py — `add_memory()` uses `datetime.now()` instead of simulated time
- Line 45: `self.memories.append(f"{datetime.now().isoformat()}: {memory}")` — this stamps memories with wall-clock time, not the simulated clock. The agent and harness use `clock.current_time` everywhere else.
- Fix: pass timestamp as a parameter, or accept it's a known inconsistency.

### plant.py — `water()` in VirtualPlant uses `datetime.now()` too
- Line 130-133: `now = datetime.now()` for watering timestamps. Same wall-clock vs simulated-clock mismatch.
- This means `watering_history` and `last_watering` are in wall-clock time while everything else in the simulation runs on `SimulatedClock`.

### plant.py — `actual_water_quantity` is set but never read
- `VirtualPlant.__init__` sets `self.actual_water_quantity` (line 112) but nothing ever uses it. Dead code.

### plant.py — action_history stores `reasoning` field, always empty now
- `add_action_to_history()` takes `reasoning` param; agent always passes `""` now. The field still gets stored in the history dict and shown in the prompt context. Harmless but noisy.

### harness.py — default model mismatch
- `harness.py` defaults to `--model llama3.2` (line 32)
- `agent.py` defaults to `model="qwen3:4b-instruct-2507-q4_K_M"` (line 57)
- These should agree. The harness default wins at runtime (it passes `args.model`), but it's confusing to have two different defaults.

### No persistence
- All state (plant hydration, memories, action history, task queue) lives in memory. Restarting the harness resets everything. No save/load mechanism.

### No error recovery in tool loop
- If a tool call fails (e.g. plant not found), the error dict is fed back to the model. The model might retry or might not. There's no explicit retry/backoff logic.

### ActualPlant is a dead stub
- `get_hydration_status()` raises `NotImplementedError`. `water()` appends to history then raises. If anyone constructs an `ActualPlant` and calls `water()`, it mutates state then crashes — partial write before exception.

## Precautions

### When editing tool definitions
The 4 stub functions at the top of `agent.py` (`take_picture`, `water`, `add_task`, `add_memory`) are the **single source of truth** for tool schemas. The ollama SDK reads their:
- **Function name** → tool name
- **Parameter names + type hints** → argument schema
- **Docstring** → tool description
- **Default values** → optional vs required

If you rename a parameter, add a parameter, or change a type hint, the schema changes automatically. But you must also update `_execute_tool()` to match — the dispatch reads `args.get("param_name")` by string.

### When adding new tools
1. Add a stub function with type hints + docstring
2. Add it to the `TOOLS` list
3. Add an `if name == "new_tool":` branch in `_execute_tool()`

### When changing the prompt template
`agent_prompt.txt` uses `str.format()` with these placeholders:
- `{plant_info}` — e.g. "Plant 0: Cactus1 (cactus)"
- `{current_time}` — ISO timestamp
- `{additional_context}` — optional context string
- `{plant_history}` — formatted memories + action history

Curly braces in the template that are NOT placeholders must be escaped as `{{` / `}}`.

### Model compatibility
Not all Ollama models support tool calling. The model must support the tool calling protocol. Known working: `llama3.2`, `qwen3`, `mistral`. If a model doesn't support tools, `msg.tool_calls` will always be empty/None and the agent will just return text without taking any actions.

### SimulatedClock vs datetime.now()
The codebase has two time domains: `clock.current_time` (simulation) and `datetime.now()` (wall-clock). These are mixed in `plant.py` (see hotspots above). Be careful when adding time-dependent logic — always ask which clock should be used.

## Common pitfalls

1. **Forgetting to set `agent.clock` and `agent.task_queue`** — these are set via direct assignment after construction. If missing, `act()` returns `{"error": "Clock not set"}` and `process_due_tasks()` returns `[]`. No crash, but silent failure.

2. **Tool function bodies look like dead code** — they are. The SDK only reads signatures. Don't add logic to the stub bodies thinking it'll execute.

3. **`messages.append(msg)` in the tool loop** — `msg` is an ollama SDK `Message` object, not a dict. The SDK accepts both. Don't accidentally try to serialize it with `json.dumps()`.

4. **`action_history` is capped at 10 entries** — `add_action_to_history()` truncates to last 10. If you're looking for full history, it's gone. Only the logger has the complete record.

5. **Hydration starts at 100** — `VirtualPlant` initializes at `hydration_level = 100.0`. The "good" range is 40-80. So a fresh plant reads as overhydrated until it decays. This is by design (simulates a freshly-watered plant) but can confuse first-time readers.

6. **`get_health_score()` optimal hydration is 50** — the score formula: `100 - abs(hydration_level - 50) * 2`. Peak health at exactly 50. This is only used for the harness status display, not by the agent's decision making.

7. **Task format convention** — tasks are strings like `"check_plant:Cactus1"`. The `process_due_tasks()` method splits on `:` to extract plant name. If the LLM generates a different format, the plant name won't be extracted and the agent will fall back to the current plant index.
