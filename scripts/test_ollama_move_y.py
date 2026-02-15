import ollama

resp = ollama.chat(
    model="qwen3:4b-instruct-2507-q4_K_M",
    messages=[{"role": "user", "content": "Move Y forward for 2 seconds."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "move_y",
            "description": "Move the Y-axis motor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["forward", "reverse"]},
                    "ms": {"type": "integer", "description": "Duration in milliseconds."},
                },
                "required": ["direction", "ms"],
            },
        },
    }],
)

tc = resp.message.tool_calls[0]
print(tc.function.name, tc.function.arguments)
