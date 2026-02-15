# Plant Care AI Agent - MCP Server

This MCP (Model Context Protocol) server wraps the Plant Care AI Agent functionality, exposing its capabilities as tools that can be accessed via the MCP protocol.

## What This Server Does

The Plant Care AI Agent is a simulation framework where an AI agent (powered by Ollama) autonomously manages virtual or real plants. This MCP server provides programmatic access to:

- Running plant care simulations
- Checking plant status and health
- Viewing agent logs and activities
- Accessing plant species information
- Testing Ollama connectivity
- Managing the simulation environment

## Server Details

- **Server Name**: Plant Care AI Agent
- **Transport**: Streamable HTTP
- **Host**: 0.0.0.0
- **Port**: 8765
- **Endpoint**: http://localhost:8765/mcp

## Available Tools

### 1. `run_simulation`
Run the plant care simulation for a specified number of steps.

**Parameters:**
- `steps` (int): Number of time steps to simulate (default: 10)
- `time_step` (int): Time step in minutes (default: 30)
- `model` (str): Ollama model to use (default: "llama3.2")

**Returns:** Simulation output including plant status updates and agent actions

### 2. `check_plant_status`
Check the current status of all plants by reading the most recent log file.

**Returns:** Latest plant status information from agent logs

### 3. `list_available_species`
List all available plant species and their care requirements.

**Returns:** JSON formatted list of plant species with watering instructions

### 4. `view_agent_prompt`
View the system prompt used by the AI agent to understand how it's instructed to care for plants.

**Returns:** The agent's system prompt

### 5. `view_tool_schema`
View the tool schema available to the AI agent, showing what actions it can take.

**Returns:** The tool schema specification

### 6. `list_logs`
List all available agent log files with timestamps.

**Returns:** List of log files sorted by date

### 7. `read_log`
Read a specific agent log file.

**Parameters:**
- `log_name` (str): Name of the log file (e.g., "agent_20260214_141737.log")

**Returns:** Contents of the log file

### 8. `test_ollama_connection`
Test connection to Ollama API and check if the model is available.

**Parameters:**
- `model` (str): Ollama model name to test (default: "llama3.2")

**Returns:** Connection status and model availability

### 9. `view_project_structure`
View the project directory structure.

**Returns:** Tree view of the project files

### 10. `install_dependencies`
Install project dependencies from requirements.txt.

**Returns:** Installation output

## Installation

1. Create a virtual environment and install dependencies:
```bash
cd /home/lethan3/Documents/code/treehacks/cactushacks/poke-mcp
uv venv .venv
uv pip install -r requirements.txt -p .venv/bin/python
```

2. Start the server:
```bash
.venv/bin/python server.py
```

Or run in the background:
```bash
.venv/bin/python server.py &
```

## Prerequisites

- Python 3.8+
- Ollama running locally (http://localhost:11434)
- Required Python packages (see requirements.txt)

## Project Structure

```
poke-mcp/
├── server.py           # FastMCP server implementation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Usage Example

The server exposes tools via the MCP protocol. Clients can connect to `http://localhost:8765/mcp` and invoke tools such as:

```python
# Example: Run a simulation
run_simulation(steps=5, time_step=30, model="llama3.2")

# Example: Check plant status
check_plant_status()

# Example: List available species
list_available_species()
```

## Related Files

The original project is located at `/home/lethan3/Documents/code/treehacks/cactushacks/` and includes:

- `agent.py` - Main AI agent implementation
- `harness.py` - Simulation harness
- `utils/` - Utility modules (plant, clock, priority queue, logger)
- `tools/` - Agent tools (take_picture, water, add_task, add_memory)
- `prompts/` - Agent prompts and tool schemas

## Notes

- The server runs on all interfaces (0.0.0.0) for accessibility
- All operations use the original project directory as working directory
- Logs are written to the main project's `logs/` directory
- The server does not modify the original project files
