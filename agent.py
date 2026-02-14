import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.plant import Plant
from utils.priority_queue import TaskQueue
from utils.clock import SimulatedClock
from utils.logger import AgentLogger
import tools.take_picture as take_picture
import tools.water as water
import tools.add_task as add_task
import tools.add_memory as add_memory


class PlantCareAgent:
    """AI agent that manages plant care using Ollama."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2",
                 log_dir: Optional[Path] = None, enable_logging: bool = True, timeout: int = 300):
        """
        Initialize the plant care agent.
        
        Args:
            ollama_url: URL of Ollama API
            model: Model name to use
            log_dir: Directory for log files (defaults to logs/ in project root)
            enable_logging: Whether to enable logging
            timeout: Timeout in seconds for Ollama API calls (default: 120)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout
        self.plants: Dict[str, Plant] = {}
        self.plants_by_index: Dict[int, Plant] = {}  # Index-based lookup
        self.task_queue: Optional[TaskQueue] = None
        self.clock: Optional[SimulatedClock] = None
        self.current_plant_index: int = 0  # Track which plant we're currently caring for
        
        # Initialize logger
        self.logger = AgentLogger(log_dir=log_dir, log_to_file=enable_logging) if enable_logging else None
        
        # Load prompt templates
        prompts_dir = Path(__file__).parent / "prompts"
        self._tool_schema = self._load_prompt(prompts_dir / "tool_schema.txt")
        self._agent_prompt_template = self._load_prompt(prompts_dir / "agent_prompt.txt")
    
    def _load_prompt(self, file_path: Path) -> str:
        """Load a prompt template from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading prompt file {file_path}: {e}")
        
    def register_plants(self, plants: List[Plant]):
        """Register plants for the agent to manage."""
        current_time = self.clock.get_current_time() if self.clock else datetime.now()
        for plant in plants:
            self.plants[plant.name] = plant
            self.plants_by_index[plant.index] = plant
            # Add initial action indicating agent started caring for this plant
            plant.add_initial_action(current_time)
    
    def set_task_queue(self, task_queue: TaskQueue):
        """Set the task queue for scheduling."""
        self.task_queue = task_queue
    
    def set_clock(self, clock: SimulatedClock):
        """Set the clock for time tracking."""
        self.clock = clock
    
    def _call_ollama(self, prompt: str) -> str:
        """Make a call to Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            return f"Error calling Ollama: Request timed out after {self.timeout} seconds. The model may be too slow or the prompt too complex. Try increasing the timeout or using a faster model."
        except requests.exceptions.ConnectionError:
            return f"Error calling Ollama: Could not connect to {self.ollama_url}. Make sure Ollama is running."
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def _get_tool_schema(self) -> str:
        """Get schema description of available tools."""
        return self._tool_schema
    
    def _get_plant_info(self, plant_name: Optional[str] = None) -> str:
        """Get basic plant information (without history)."""
        if not self.plants:
            return "No plants registered."
        
        # Determine which plant to show
        if plant_name and plant_name in self.plants:
            plant = self.plants[plant_name]
        elif self.current_plant_index in self.plants_by_index:
            plant = self.plants_by_index[self.current_plant_index]
        else:
            # Fallback to first plant
            plant = list(self.plants.values())[0]
        
        return f"Plant {plant.index}: {plant.name} ({plant.species})"
    
    def _get_plant_history(self, plant_name: Optional[str] = None) -> str:
        """Get plant history (to be shown at bottom of prompt)."""
        if not self.plants:
            return ""
        
        # Determine which plant to show
        if plant_name and plant_name in self.plants:
            plant = self.plants[plant_name]
        elif self.current_plant_index in self.plants_by_index:
            plant = self.plants_by_index[self.current_plant_index]
        else:
            # Fallback to first plant
            plant = list(self.plants.values())[0]
        
        context = "\n" + "="*60 + "\n"
        context += "Plant History (automatically included):\n"
        context += "="*60 + "\n\n"
        
        # ALWAYS include memories (automatically provided, no need to fetch)
        context += "Plant Memories:\n"
        if plant.memories:
            for memory in plant.memories[-5:]:  # Last 5 memories
                context += f"  - {memory}\n"
        else:
            context += f"  (No memories yet)\n"
        
        # ALWAYS include recent action history (automatically provided, no need to fetch)
        context += "\nRecent Action History:\n"
        recent_history = plant.get_recent_history(max_items=5)
        if recent_history:
            for action in recent_history:
                tool = action.get("tool", "unknown")
                timestamp = action.get("timestamp", "")
                # Show just the timestamp (time part) for brevity
                time_str = timestamp.split("T")[1].split(".")[0] if "T" in timestamp else timestamp
                args = action.get("args", {})
                result = action.get("result", {})
                
                # Format the action without reasoning, showing full tool call and result
                context += f"  - [{time_str}] {tool}"
                if args:
                    # Show args in a readable format
                    args_str = json.dumps(args, indent=2).replace('\n', '\n    ')
                    context += f" with args: {args_str}\n"
                else:
                    context += "\n"
                
                # Show full result if available
                if result:
                    if isinstance(result, dict):
                        result_str = json.dumps(result, indent=2).replace('\n', '\n    ')
                        context += f"    Result: {result_str}\n"
                    else:
                        context += f"    Result: {result}\n"
        else:
            context += f"  (No previous actions)\n"
        
        return context
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        reasoning = tool_call.get("reasoning", "No reasoning provided")
        
        current_time = self.clock.get_current_time() if self.clock else datetime.now()
        
        # Log tool call
        if self.logger:
            self.logger.log_tool_call(tool_call, current_time)
        
        result = None
        plant_name = None
        
        if tool_name == "take_picture":
            plant_name = args.get("plant_name")
            if plant_name not in self.plants:
                result = {"error": f"Plant '{plant_name}' not found"}
            else:
                result = take_picture.take_picture(self.plants[plant_name])
                # Log plant status
                if self.logger:
                    self.logger.log_plant_status(plant_name, result, current_time)
                # Add to plant history
                self.plants[plant_name].add_action_to_history(tool_name, args, reasoning, result, current_time)
        
        elif tool_name == "water":
            plant_name = args.get("plant_name")
            water_quantity = args.get("water_quantity", 1.0)
            if plant_name not in self.plants:
                result = {"error": f"Plant '{plant_name}' not found"}
            else:
                result = water.water(self.plants[plant_name], water_quantity)
                # Add to plant history
                self.plants[plant_name].add_action_to_history(tool_name, args, reasoning, result, current_time)
        
        elif tool_name == "add_task":
            if not self.task_queue or not self.clock:
                result = {"error": "Task queue or clock not set"}
            else:
                task_desc = args.get("task_description", "")
                priority = args.get("priority", 5.0)
                minutes = args.get("minutes", 60.0)
                result = add_task.add_task(
                    self.task_queue, task_desc, priority, minutes, self.clock.get_current_time()
                )
                # Tasks are global, not plant-specific, so we don't add to plant history
        
        elif tool_name == "add_memory":
            plant_name = args.get("plant_name")
            memory = args.get("memory", "")
            if plant_name not in self.plants:
                result = {"error": f"Plant '{plant_name}' not found"}
            else:
                result = add_memory.add_memory(self.plants[plant_name], memory)
                # Add to plant history
                self.plants[plant_name].add_action_to_history(tool_name, args, reasoning, result, current_time)
        
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        # Log tool result
        if self.logger:
            self.logger.log_tool_result(tool_call, result, current_time)
        
        return result
    
    def act(self, context: Optional[str] = None, plant_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Activate the agent to make decisions and call tools for a specific plant.
        
        Args:
            context: Optional additional context for the agent
            plant_name: Name of the plant to care for (if None, uses current_plant_index)
            
        Returns:
            Dictionary with agent's actions and results
        """
        if not self.clock:
            return {"error": "Clock not set"}
        
        current_time = self.clock.get_current_time()
        
        # Determine which plant to care for
        if plant_name and plant_name in self.plants:
            target_plant = self.plants[plant_name]
            self.current_plant_index = target_plant.index
        elif self.current_plant_index in self.plants_by_index:
            target_plant = self.plants_by_index[self.current_plant_index]
        else:
            # Fallback to first plant
            target_plant = list(self.plants.values())[0]
            self.current_plant_index = target_plant.index
        
        plant_info = self._get_plant_info(target_plant.name)
        plant_history = self._get_plant_history(target_plant.name)
        
        # Log agent activation
        if self.logger:
            self.logger.log_agent_activation(context, plant_info + plant_history, current_time)
        
        # Build prompt from template
        prompt = self._agent_prompt_template.format(
            plant_info=plant_info,
            current_time=current_time.isoformat(),
            tool_schema=self._get_tool_schema(),
            additional_context=context or "",
            plant_history=plant_history
        )
        
        # Log prompt
        if self.logger:
            self.logger.log_prompt(prompt, current_time)
        
        # Call Ollama
        response = self._call_ollama(prompt)
        
        # Log response
        if self.logger:
            self.logger.log_response(response, current_time)
        
        # Try to parse tool calls from response
        tool_results = []
        reasoning = None
        try:
            # Try to extract JSON from response
            if "{" in response:
                import re
                # Try to parse the full response as JSON first (handles nested arrays better)
                try:
                    # Look for JSON object that might span multiple lines
                    # Try to find complete JSON objects including arrays
                    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                    matches = re.finditer(json_pattern, response, re.DOTALL)
                    
                    for match in matches:
                        try:
                            parsed = json.loads(match.group())
                            if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                                # New format: reasoning outside, tool_calls array
                                reasoning = parsed.get("reasoning", "No reasoning provided")
                                tool_calls = parsed["tool_calls"]
                                
                                for tool_call in tool_calls:
                                    if "tool" in tool_call:
                                        # Add reasoning to each tool call for execution
                                        tool_call_with_reasoning = tool_call.copy()
                                        tool_call_with_reasoning["reasoning"] = reasoning
                                        result = self._execute_tool(tool_call_with_reasoning)
                                        tool_results.append({
                                            "tool_call": tool_call,
                                            "result": result
                                        })
                                break  # Found the main response, stop looking
                        except json.JSONDecodeError:
                            continue
                except Exception:
                    pass
                
                # If we didn't find the new format, try simpler pattern matching
                if not tool_results:
                    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
                    for match in json_matches:
                        try:
                            parsed = json.loads(match)
                            if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                                # New format: reasoning outside, tool_calls array
                                reasoning = parsed.get("reasoning", "No reasoning provided")
                                tool_calls = parsed["tool_calls"]
                                
                                for tool_call in tool_calls:
                                    if "tool" in tool_call:
                                        tool_call_with_reasoning = tool_call.copy()
                                        tool_call_with_reasoning["reasoning"] = reasoning
                                        result = self._execute_tool(tool_call_with_reasoning)
                                        tool_results.append({
                                            "tool_call": tool_call,
                                            "result": result
                                        })
                                break
                            elif "tool" in parsed and "reasoning" in parsed:
                                # Old format: single tool call with reasoning inside (backward compatibility)
                                result = self._execute_tool(parsed)
                                tool_results.append({
                                    "tool_call": parsed,
                                    "result": result
                                })
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            pass
        
        return {
            "response": response,
            "tool_results": tool_results,
            "reasoning": reasoning,
            "timestamp": current_time.isoformat()
        }
    
    def process_due_tasks(self) -> List[Dict[str, Any]]:
        """Process all due tasks from the task queue."""
        if not self.task_queue or not self.clock:
            return []
        
        current_time = self.clock.get_current_time()
        due_tasks = self.task_queue.get_all_due(current_time)
        
        results = []
        for scheduled_time, priority, task in due_tasks:
            # Task format: "check_plant:PlantName" or just plant name
            plant_name = None
            if isinstance(task, str):
                if ":" in task:
                    # Format: "action:plant_name"
                    parts = task.split(":", 1)
                    plant_name = parts[1] if len(parts) > 1 else None
                else:
                    # Assume it's a plant name
                    plant_name = task if task in self.plants else None
            
            # Activate agent for the specific plant
            result = self.act(context=f"Scheduled task: {task}", plant_name=plant_name)
            results.append({
                "task": task,
                "scheduled_time": scheduled_time.isoformat(),
                "priority": priority,
                "action_result": result
            })
        
        return results
    
    def get_log_file(self) -> Optional[Path]:
        """Get the path to the current log file."""
        if self.logger:
            return self.logger.log_file
        return None
    
    def print_log_summary(self):
        """Print a summary of the agent's log."""
        if self.logger:
            self.logger.print_summary()