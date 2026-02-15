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
- **Host**: 0.0.0.0 (configurable via HOST env var)
- **Port**: 8000 (configurable via PORT env var, defaults to 8000)
- **Endpoint**: http://localhost:8000/mcp (NOTE: Must include `/mcp`!)

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
cd poke-mcp
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the server:
```bash
python server.py
```

The server will start on `http://localhost:8000/mcp`

## Testing Locally

You can test the server using the MCP Inspector:

```bash
# In one terminal, start the server:
python server.py

# In another terminal, start the inspector:
npx @modelcontextprotocol/inspector
```

Then open http://localhost:3000 and connect to `http://localhost:8000/mcp` using "Streamable HTTP" transport.

**IMPORTANT**: Make sure to include `/mcp` in the URL!

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

## Deployment

### Option 1: Deploy to Render (One-Click)

1. Fork this repository
2. Connect your GitHub account to [Render](https://render.com)
3. Create a new Web Service on Render
4. Connect your forked repository
5. Render will automatically detect the `render.yaml` configuration

Your server will be available at `https://your-service-name.onrender.com/mcp` (NOTE: Include `/mcp`!)

### Option 2: Manual Deployment

1. Fork this repository
2. Connect your GitHub account to Render
3. Create a new Web Service on Render
4. Connect your forked repository
5. Set environment variables if needed:
   - `PROJECT_DIR`: Path to project directory (defaults to relative path)
   - `PORT`: Server port (defaults to 8000)

## Poke Setup

You can connect your MCP server to Poke at [poke.com/settings/connections](https://poke.com/settings/connections). 

To test the connection explicitly, ask Poke something like:
```
Tell the subagent to use the "{connection name}" integration's "{tool name}" tool
```

If you run into persistent issues of Poke not calling the right MCP (e.g. after you've renamed the connection), you may send `clearhistory` to Poke to delete all message history and start fresh.

## Usage Example

The server exposes tools via the MCP protocol. Clients can connect to `http://localhost:8000/mcp` and invoke tools such as:

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
