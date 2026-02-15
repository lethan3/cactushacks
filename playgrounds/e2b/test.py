# pip install ollama
import ollama
import subprocess
import tempfile
import os

# Send the prompt to the model
response = ollama.chat(
    model="qwen3:4b-instruct-2507-q4_K_M",
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant that can execute python code. Only respond with the code to be executed and nothing else. Strip backticks in code blocks."
    },
    {
        "role": "user",
        "content": "Calculate how many r's are in the word 'strawberry'"
    }
])

# Extract the code from the response
code = response['message']['content']
print(f"--- Generated code ---\n{code}\n--- End code ---\n")

# Execute code in a local subprocess sandbox
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(code)
    tmp_path = f.name

try:
    result = subprocess.run(
        ['python', tmp_path],
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
finally:
    os.unlink(tmp_path)