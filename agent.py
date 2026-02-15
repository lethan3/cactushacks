import json
import ollama
from pathlib import Path
from datetime import datetime, timedelta

from utils.logger import AgentLogger
from utils.poke_ai import PokeAIClient


# ---------------------------------------------------------------------------
# Tool definitions — ollama SDK reads signatures + docstrings to auto-generate
# JSON schemas. These bodies are never called by the SDK.
# ---------------------------------------------------------------------------

def take_picture(plant_name: str) -> str:
    """Take a picture of a plant and return its current status.
    Returns hydration level, watering frequency, and species care instructions.

    Args:
        plant_name: Name of the plant to photograph.
    """


def water(plant_name: str, water_quantity: float = 1.0) -> str:
    """Water a plant with the specified quantity.

    Args:
        plant_name: Name of the plant to water.
        water_quantity: Amount of water in cups.
    """


def add_task(task_description: str, priority: float = 5.0, minutes: float = 60.0) -> str:
    """Schedule a future task for plant care.

    Args:
        task_description: What the task is, e.g. 'check_plant:Cactus1'.
        priority: Lower number means higher priority.
        minutes: Minutes from now to run the task.
    """


def add_memory(plant_name: str, memory: str) -> str:
    """Store an observation or note about a plant for future reference.

    Args:
        plant_name: Name of the plant this memory is about.
        memory: The observation or note to remember.
    """


TOOLS = [take_picture, water, add_task, add_memory]


class PlantCareAgent:
    """AI agent that manages plant care using Ollama with native tool calling."""

    def __init__(self, model="qwen3:4b-instruct-2507-q4_K_M", host="http://localhost:11434",
                 log_dir=None, enable_logging=True, timeout=300, max_rounds=5,
                 poke_ai_api_key=None, poke_ai_api_url=None, poke_ai_enabled=True):
        self.model = model
        self.client = ollama.Client(host=host, timeout=timeout)
        self.max_rounds = max_rounds
        self.plants = {}
        self.plants_by_index = {}
        self.task_queue = None
        self.clock = None
        self.current_plant_index = 0

        self.logger = AgentLogger(log_dir=log_dir, log_to_file=enable_logging) if enable_logging else None
        self.poke_ai = PokeAIClient(
            api_key=poke_ai_api_key,
            api_url=poke_ai_api_url,
            enabled=poke_ai_enabled
        )
        self._system_prompt = (Path(__file__).parent / "prompts" / "agent_prompt.txt").read_text().strip()

    def register_plants(self, plants):
        now = self.clock.current_time if self.clock else datetime.now()
        for plant in plants:
            self.plants[plant.name] = plant
            self.plants_by_index[plant.index] = plant
            plant.add_initial_action(now)

    def _get_plant(self, name=None):
        if name and name in self.plants:
            return self.plants[name]
        if self.current_plant_index in self.plants_by_index:
            return self.plants_by_index[self.current_plant_index]
        return list(self.plants.values())[0]

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _build_system_msg(self, plant, context=None):
        """Build system prompt with plant info and history."""
        plant_info = f"Plant {plant.index}: {plant.name} ({plant.species})"
        now = self.clock.current_time

        history = ""
        if plant.memories:
            history += "Plant Memories:\n"
            for m in plant.memories[-5:]:
                history += f"  - {m}\n"

        if plant.action_history:
            history += "\nRecent Actions:\n"
            for action in plant.action_history[-5:]:
                tool = action.get("tool", "unknown")
                ts = action.get("timestamp", "")
                time_str = ts.split("T")[1].split(".")[0] if "T" in ts else ts
                args = action.get("args", {})
                result = action.get("result", {})
                history += f"  - [{time_str}] {tool}"
                if args:
                    history += f"({json.dumps(args)})"
                if result:
                    history += f" -> {json.dumps(result)}"
                history += "\n"

        return self._system_prompt.format(
            plant_info=plant_info,
            current_time=now.isoformat(),
            additional_context=context or "",
            plant_history=history,
        )

    # ------------------------------------------------------------------
    # Tool execution — dispatch by name, return result dicts
    # ------------------------------------------------------------------

    def _execute_tool(self, name, args):
        now = self.clock.current_time

        if name == "take_picture":
            plant = self.plants.get(args.get("plant_name"))
            if not plant:
                return {"error": f"Plant '{args.get('plant_name')}' not found"}
            # Pass camera and overshoot_client if available
            camera = getattr(self, 'camera', None)
            overshoot_client = getattr(self, 'overshoot_client', None)
            result = plant.take_picture(camera=camera, overshoot_client=overshoot_client)
            if self.logger:
                self.logger.log_plant_status(plant.name, result, now)
            plant.add_action_to_history(name, args, "", result, now)
            return result

        if name == "water":
            plant = self.plants.get(args.get("plant_name"))
            qty = args.get("water_quantity", 1.0)
            if not plant:
                return {"error": f"Plant '{args.get('plant_name')}' not found"}
            plant.water(qty)
            result = {"success": True, "message": f"Watered {plant.name} with {qty} cup(s)"}
            plant.add_action_to_history(name, args, "", result, now)
            return result

        if name == "add_task":
            if not self.task_queue:
                return {"error": "Task queue not set"}
            desc = args.get("task_description", "")
            priority = args.get("priority", 5.0)
            mins = args.get("minutes", 60.0)
            self.task_queue.add_task_in(desc, priority, mins, now)
            scheduled = now + timedelta(minutes=mins)
            return {"success": True, "message": f"Task '{desc}' scheduled for {scheduled.isoformat()}"}

        if name == "add_memory":
            plant = self.plants.get(args.get("plant_name"))
            memory = args.get("memory", "")
            if not plant:
                return {"error": f"Plant '{args.get('plant_name')}' not found"}
            plant.add_memory(memory)
            result = {"success": True, "message": f"Added memory about {plant.name}"}
            plant.add_action_to_history(name, args, "", result, now)
            return result

        return {"error": f"Unknown tool: {name}"}

    # ------------------------------------------------------------------
    # Agent loop — native tool calling via ollama SDK
    # ------------------------------------------------------------------

    def act(self, context=None, plant_name=None):
        """Activate the agent for a plant. Uses native Ollama tool calling."""
        if not self.clock:
            return {"error": "Clock not set"}

        now = self.clock.current_time
        plant = self._get_plant(plant_name)
        self.current_plant_index = plant.index

        system_msg = self._build_system_msg(plant, context)

        if self.logger:
            self.logger.log_agent_activation(context, system_msg, now)
            self.logger.log_prompt(system_msg, now)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Check on {plant.name} and take appropriate care actions."},
        ]

        tool_results = []
        response_text = ""

        for _ in range(self.max_rounds):
            try:
                resp = self.client.chat(model=self.model, 
                                        messages=messages,
                                        tools=TOOLS,
                                        think=False)
            except Exception as e:
                response_text = f"Error: {e}"
                break

            msg = resp.message
            response_text = msg.content or ""

            if self.logger and response_text:
                self.logger.log_response(response_text, now)

            # No tool calls → model is done
            if not msg.tool_calls:
                break

            # Add assistant message (with tool calls) to conversation
            messages.append(msg)

            # Execute each tool call and feed results back
            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments
                tc_dict = {"tool": name, "args": args}

                if self.logger:
                    self.logger.log_tool_call(tc_dict, now)

                result = self._execute_tool(name, args)
                tool_results.append({"tool": name, "args": args, "result": result})

                if self.logger:
                    self.logger.log_tool_result(tc_dict, result, now)

                messages.append({"role": "tool", "content": json.dumps(result)})

        result_dict = {
            "response": response_text,
            "tool_results": tool_results,
            "timestamp": now.isoformat(),
        }
        
        # Send message to Poke AI after response
        if self.poke_ai.enabled:
            message_content = self.poke_ai.format_agent_response(
                response_text=response_text,
                tool_results=tool_results,
                plant_name=plant.name,
                timestamp=now.isoformat()
            )
            metadata = {
                "plant_name": plant.name,
                "plant_species": plant.species,
                "timestamp": now.isoformat(),
                "tool_count": len(tool_results)
            }
            self.poke_ai.send_message(message_content, metadata)
        
        return result_dict

    def process_due_tasks(self):
        if not self.task_queue or not self.clock:
            return []

        due = self.task_queue.get_all_due(self.clock.current_time)

        results = []
        for scheduled_time, priority, task in due:
            plant_name = None
            if isinstance(task, str):
                if ":" in task:
                    plant_name = task.split(":", 1)[1]
                elif task in self.plants:
                    plant_name = task

            result = self.act(context=f"Scheduled task: {task}", plant_name=plant_name)
            results.append({
                "task": task,
                "scheduled_time": scheduled_time.isoformat(),
                "priority": priority,
                "action_result": result,
            })

        return results

    def get_log_file(self):
        return self.logger.log_file if self.logger else None

    def print_log_summary(self):
        if self.logger:
            self.logger.print_summary()
