#!/usr/bin/env python3
"""
FastMCP server wrapping the Plant Care AI Agent functionality.
Based on https://github.com/InteractionCo/mcp-server-template
"""
import subprocess
import sys
import json
import os
from pathlib import Path
from fastmcp import FastMCP

# Project directory - use relative path or environment variable
PROJECT_DIR = os.getenv("PROJECT_DIR", str(Path(__file__).parent.parent))

mcp = FastMCP("Plant Care AI Agent")


@mcp.tool()
def run_simulation(steps: int = 10, time_step: int = 30, model: str = "llama3.2") -> str:
    """
    Run the plant care simulation for a specified number of steps.

    Args:
        steps: Number of time steps to simulate (default: 10)
        time_step: Time step in minutes (default: 30)
        model: Ollama model to use (default: llama3.2)

    Returns:
        Simulation output including plant status updates and agent actions
    """
    try:
        harness_path = Path(PROJECT_DIR) / "harness.py"
        if not harness_path.exists():
            return f"Error: harness.py not found at {harness_path}"
        
        result = subprocess.run(
            [sys.executable, str(harness_path), "--steps", str(steps), "--time-step", str(time_step), "--model", model],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=600
        )

        output = result.stdout
        if result.stderr:
            output += f"\n\nSTDERR:\n{result.stderr}"

        if result.returncode != 0:
            output += f"\n\nCommand failed with return code: {result.returncode}"

        return output
    except subprocess.TimeoutExpired:
        return "Error: Simulation timed out after 10 minutes"
    except Exception as e:
        return f"Error running simulation: {str(e)}"


@mcp.tool()
def check_plant_status() -> str:
    """
    Check the current status of all plants in the simulation.
    This reads the most recent log file to see agent activities.

    Returns:
        Latest plant status information from agent logs
    """
    try:
        logs_dir = Path(PROJECT_DIR) / "logs"
        if not logs_dir.exists():
            return "No logs directory found. Run a simulation first."

        # Get most recent log file
        log_files = sorted(logs_dir.glob("agent_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not log_files:
            return "No log files found. Run a simulation first."

        latest_log = log_files[0]

        # Read last 100 lines of the log
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                last_lines = lines[-100:] if len(lines) > 100 else lines
                return f"Latest log file: {latest_log.name}\n\n{''.join(last_lines)}"
        except Exception as e:
            return f"Error reading log file: {str(e)}"

        return f"Latest log file: {latest_log.name}\n\n{result.stdout}"
    except Exception as e:
        return f"Error reading plant status: {str(e)}"


@mcp.tool()
def list_available_species() -> str:
    """
    List all available plant species and their care requirements.

    Returns:
        JSON formatted list of plant species with watering instructions
    """
    try:
        # Run Python to import and access the Plant class constants
        code = f"""
import sys
sys.path.insert(0, r'{PROJECT_DIR}')
from utils.plant import Plant
import json
print(json.dumps(Plant.SPECIES_CARE, indent=2))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR
        )

        if result.returncode == 0:
            return f"Available plant species:\n\n{result.stdout}"
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error listing species: {str(e)}"


@mcp.tool()
def view_agent_prompt() -> str:
    """
    View the system prompt used by the AI agent.
    This shows how the agent is instructed to care for plants.

    Returns:
        The agent's system prompt
    """
    try:
        prompt_file = Path(PROJECT_DIR) / "prompts" / "agent_prompt.txt"

        if not prompt_file.exists():
            return "Agent prompt file not found"

        with open(prompt_file, 'r') as f:
            content = f.read()

        return content
    except Exception as e:
        return f"Error reading agent prompt: {str(e)}"


@mcp.tool()
def view_tool_schema() -> str:
    """
    View the tool schema available to the AI agent.
    This shows what actions the agent can take.

    Returns:
        The tool schema specification
    """
    try:
        schema_file = Path(PROJECT_DIR) / "prompts" / "tool_schema.txt"

        if not schema_file.exists():
            return "Tool schema file not found"

        with open(schema_file, 'r') as f:
            content = f.read()

        return content
    except Exception as e:
        return f"Error reading tool schema: {str(e)}"


@mcp.tool()
def list_logs() -> str:
    """
    List all available agent log files with timestamps.

    Returns:
        List of log files sorted by date
    """
    try:
        logs_dir = Path(PROJECT_DIR) / "logs"

        if not logs_dir.exists():
            return "No logs directory found"

        log_files = sorted(logs_dir.glob("agent_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not log_files:
            return "No log files found"

        output = "Available log files (most recent first):\n\n"
        for log_file in log_files:
            stat = log_file.stat()
            from datetime import datetime
            mtime = datetime.fromtimestamp(stat.st_mtime)
            size = stat.st_size
            output += f"- {log_file.name} ({size} bytes, modified: {mtime})\n"

        return output
    except Exception as e:
        return f"Error listing logs: {str(e)}"


@mcp.tool()
def read_log(log_name: str) -> str:
    """
    Read a specific agent log file.

    Args:
        log_name: Name of the log file (e.g., agent_20260214_141737.log)

    Returns:
        Contents of the log file
    """
    try:
        log_file = Path(PROJECT_DIR) / "logs" / log_name

        if not log_file.exists():
            return f"Log file not found: {log_name}"

        with open(log_file, 'r') as f:
            content = f.read()

        return content
    except Exception as e:
        return f"Error reading log: {str(e)}"


@mcp.tool()
def test_ollama_connection(model: str = "llama3.2") -> str:
    """
    Test connection to Ollama API and check if the model is available.

    Args:
        model: Ollama model name to test (default: llama3.2)

    Returns:
        Connection status and model availability
    """
    try:
        # Test with a simple Python script
        code = f"""
import requests
import json

try:
    # Test connection
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        models = response.json().get('models', [])
        model_names = [m.get('name', '') for m in models]

        print(f"Ollama is running. Available models: {{', '.join(model_names)}}")

        # Check if requested model is available
        if '{model}' in model_names or any('{model}' in name for name in model_names):
            print(f"\\nModel '{model}' is available.")
        else:
            print(f"\\nWarning: Model '{model}' not found. You may need to pull it with: ollama pull {model}")
    else:
        print(f"Ollama API returned status code: {{response.status_code}}")
except requests.exceptions.ConnectionError:
    print("Error: Cannot connect to Ollama at http://localhost:11434")
    print("Make sure Ollama is running with: ollama serve")
except Exception as e:
    print(f"Error: {{str(e)}}")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=10
        )

        return result.stdout + result.stderr
    except Exception as e:
        return f"Error testing Ollama connection: {str(e)}"


@mcp.tool()
def view_project_structure() -> str:
    """
    View the project directory structure.

    Returns:
        Tree view of the project files
    """
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.py", "-o", "-name", "*.txt", "-o", "-name", "*.md"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR
        )

        files = sorted(result.stdout.strip().split('\n'))

        output = "Project Structure:\n\n"
        for file in files:
            if not file.startswith('./.git') and not file.startswith('./poke-mcp'):
                output += f"{file}\n"

        return output
    except Exception as e:
        return f"Error viewing project structure: {str(e)}"


@mcp.tool()
def install_dependencies() -> str:
    """
    Install project dependencies from requirements.txt.

    Returns:
        Installation output
    """
    try:
        requirements_path = Path(PROJECT_DIR) / "requirements.txt"
        if not requirements_path.exists():
            return f"Error: requirements.txt not found at {requirements_path}"
        
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=300
        )

        output = result.stdout
        if result.stderr:
            output += f"\n\nSTDERR:\n{result.stderr}"

        if result.returncode == 0:
            output += "\n\nDependencies installed successfully!"
        else:
            output += f"\n\nInstallation failed with return code: {result.returncode}"

        return output
    except Exception as e:
        return f"Error installing dependencies: {str(e)}"


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("Starting Plant Care AI Agent MCP Server...")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Server running on http://{host}:{port}/mcp")
    print("NOTE: Connect to the /mcp endpoint using 'Streamable HTTP' transport")
    
    mcp.run(transport="streamable-http", host=host, port=port)
