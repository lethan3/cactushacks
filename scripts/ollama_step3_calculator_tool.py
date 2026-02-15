"""Step 3: tool calling — calculator with qwen3:4b, with timing."""
import re
import ollama

MODEL = "qwen3:4b"

def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def safe_calc(expr):
    """Eval arithmetic only. Allowlist: digits, operators, parens, whitespace."""
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%]+$', expr):
        raise ValueError(f"Blocked expression: {expr!r}")
    return str(eval(expr))  # safe: only math chars pass the regex

def print_timing(resp):
    """Print timing breakdown from Ollama's built-in nanosecond fields."""
    ns = 1e9
    total     = resp.total_duration / ns
    load      = resp.load_duration / ns
    prompt    = resp.prompt_eval_duration / ns
    gen       = resp.eval_duration / ns
    p_tokens  = resp.prompt_eval_count
    g_tokens  = resp.eval_count

    ttft = total - gen  # everything before first generated token
    gen_tps = g_tokens / gen if gen > 0 else 0
    prompt_tps = p_tokens / prompt if prompt > 0 else 0

    print(f"  total        : {total:.2f}s")
    print(f"  TTFT         : {ttft:.2f}s  (load {load:.2f}s + prompt {prompt:.2f}s + overhead)")
    print(f"  generation   : {gen:.2f}s")
    print(f"  prompt tokens: {p_tokens}  ({prompt_tps:.1f} tok/s)")
    print(f"  gen tokens   : {g_tokens}  ({gen_tps:.1f} tok/s)")

# --- tool definition (OpenAI-compatible schema) ---
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Returns the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '2 + 2' or '(3 * 5) / 2'",
                }
            },
            "required": ["expression"],
        },
    },
}

# --- round 1: ask a question that requires the tool ---
messages = [
    {"role": "user", "content": "What is 78901 * 4567 + 123? Use the calculator tool."},
]

print("=== Round 1: sending user message ===")
resp1 = ollama.chat(model=MODEL, messages=messages, tools=[calculator_tool])

print(f"  content : {resp1.message.content!r}")
print(f"  tool_calls: {resp1.message.tool_calls}")
print_timing(resp1)

if not resp1.message.tool_calls:
    print("ERROR: model did not produce a tool call. Raw response above.")
    raise SystemExit(1)

# --- execute the tool call ---
tc = resp1.message.tool_calls[0]
fn_name = tc.function.name
fn_args = tc.function.arguments  # dict
expr = fn_args.get("expression", "")

print(f"\n=== Tool call: {fn_name}({fn_args}) ===")
result = safe_calc(expr)
print(f"  result: {result}")

# --- round 2: send tool result back, get final answer ---
# Important: include the assistant's tool-call message verbatim so the model
# sees the full conversation. resp1.message is already the right object.
messages.append(resp1.message)
messages.append({"role": "tool", "content": result})

print("\n=== Round 2: sending tool result ===")
resp2 = ollama.chat(model=MODEL, messages=messages, tools=[calculator_tool])

answer = strip_think(resp2.message.content or "")
print(f"  raw    : {resp2.message.content!r}")
print(f"  answer : {answer}")
print_timing(resp2)

# --- verify ---
expected = 78901 * 4567 + 123
print(f"\n=== Verification ===")
print(f"  expected: {expected}")
print(f"  correct : {str(expected) in answer}")
