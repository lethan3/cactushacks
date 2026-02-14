import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class AgentLogger:
    """Logger for tracking agent inputs and outputs."""
    
    def __init__(self, log_dir: Optional[Path] = None, log_to_file: bool = True):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save log files (defaults to logs/ in project root)
            log_to_file: Whether to write logs to file
        """
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        self.log_dir = log_dir
        self.log_to_file = log_to_file
        
        if self.log_to_file:
            self.log_dir.mkdir(exist_ok=True)
            # Create a new log file for this session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"agent_{timestamp}.log"
            self.session_log = []
    
    def log_agent_activation(self, context: Optional[str], plants_context: str, 
                            current_time: datetime):
        """Log when agent is activated."""
        entry = {
            "type": "agent_activation",
            "timestamp": current_time.isoformat(),
            "context": context,
            "plants_context": plants_context
        }
        self._write_entry(entry)
    
    def log_prompt(self, prompt: str, current_time: datetime):
        """Log the prompt sent to Ollama."""
        entry = {
            "type": "prompt",
            "timestamp": current_time.isoformat(),
            "prompt": prompt
        }
        self._write_entry(entry)
    
    def log_response(self, response: str, current_time: datetime):
        """Log the response from Ollama."""
        entry = {
            "type": "response",
            "timestamp": current_time.isoformat(),
            "response": response
        }
        self._write_entry(entry)
    
    def log_tool_call(self, tool_call: Dict[str, Any], current_time: datetime):
        """Log a tool call."""
        entry = {
            "type": "tool_call",
            "timestamp": current_time.isoformat(),
            "tool": tool_call.get("tool"),
            "args": tool_call.get("args", {}),
            "reasoning": tool_call.get("reasoning", "No reasoning provided")
        }
        self._write_entry(entry)
    
    def log_tool_result(self, tool_call: Dict[str, Any], result: Dict[str, Any], 
                       current_time: datetime):
        """Log the result of a tool call."""
        entry = {
            "type": "tool_result",
            "timestamp": current_time.isoformat(),
            "tool": tool_call.get("tool"),
            "reasoning": tool_call.get("reasoning", "No reasoning provided"),
            "result": result
        }
        self._write_entry(entry)
    
    def log_plant_status(self, plant_name: str, status: Dict[str, Any], 
                        current_time: datetime):
        """Log plant status information."""
        entry = {
            "type": "plant_status",
            "timestamp": current_time.isoformat(),
            "plant_name": plant_name,
            "status": status
        }
        self._write_entry(entry)
    
    def log_error(self, error_message: str):
        """Log an error message."""
        entry = {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        self._write_entry(entry)
    
    def _write_entry(self, entry: Dict[str, Any]):
        """Write a log entry."""
        if self.log_to_file:
            self.session_log.append(entry)
            # Write to file immediately
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, indent=2))
                f.write("\n" + "-" * 80 + "\n")
    
    def get_session_log(self) -> list:
        """Get all log entries for this session."""
        return self.session_log
    
    def print_summary(self):
        """Print a summary of the session."""
        if not self.session_log:
            print("No log entries.")
            return
        
        print(f"\n{'='*80}")
        print(f"Agent Log Summary")
        print(f"{'='*80}")
        print(f"Log file: {self.log_file}")
        print(f"Total entries: {len(self.session_log)}")
        
        # Count by type
        type_counts = {}
        for entry in self.session_log:
            entry_type = entry.get("type", "unknown")
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
        
        print("\nEntry types:")
        for entry_type, count in type_counts.items():
            print(f"  {entry_type}: {count}")
        print(f"{'='*80}\n")
