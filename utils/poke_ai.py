"""Poke AI integration for sending messages when the agent responds."""
import requests
from typing import Optional, Dict, Any


class PokeAIClient:
    """Client for sending messages to Poke AI."""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, 
                 enabled: bool = True):
        """
        Initialize Poke AI client.
        
        Args:
            api_key: API key for authentication (can also be set via POKE_AI_API_KEY env var)
            api_url: API endpoint URL (default: http://0.0.0.0:8000/mcp)
            enabled: Whether to enable message sending (default: True)
        """
        import os
        self.api_key = api_key or os.getenv("POKE_AI_API_KEY")
        # URL can be set via parameter, environment variable, or defaults to the standard endpoint
        self.api_url = api_url or os.getenv("POKE_AI_API_URL", "http://0.0.0.0:8000/mcp")
        self.enabled = enabled and bool(self.api_key)
        
        # Debug: Print configuration on initialization
        if self.enabled:
            print(f"Poke AI configured: URL={self.api_url}, API key={'*' * (len(self.api_key) - 4) + self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else 'set'}")
    
    def send_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message to Poke AI.
        
        Args:
            content: Message content to send
            metadata: Optional metadata to include (plant name, timestamp, etc.)
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        if not self.api_key:
            print("Warning: Poke AI API key not set. Set POKE_AI_API_KEY env var or pass api_key parameter.")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "message": content,
                "metadata": metadata or {}
            }
            
            # Debug: Show URL being used (first time only)
            if not hasattr(self, '_url_logged'):
                print(f"Poke AI: Sending to URL: {self.api_url}")
                self._url_logged = True
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=5
            )
            
            # Check for authorization errors and provide helpful feedback
            if response.status_code == 401:
                print(f"Warning: Poke AI authentication failed (401 Unauthorized). "
                      f"Check that your API key is correct. URL: {self.api_url}")
                try:
                    error_detail = response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response: {response.text[:200]}")
                return False
            
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            # More detailed error reporting
            if hasattr(e, 'response') and e.response is not None:
                print(f"Warning: Failed to send Poke AI message: {e}")
                print(f"Status code: {e.response.status_code}")
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response: {e.response.text[:200]}")
            else:
                print(f"Warning: Failed to send Poke AI message: {e}")
            return False
        except Exception as e:
            print(f"Warning: Unexpected error sending Poke AI message: {e}")
            return False
    
    def format_agent_response(self, response_text: str, tool_results: list, 
                              plant_name: str, timestamp: str) -> str:
        """
        Format agent response for Poke AI message.
        
        Args:
            response_text: The agent's response text
            tool_results: List of tool execution results
            plant_name: Name of the plant being cared for
            timestamp: ISO timestamp of the response
            
        Returns:
            Formatted message string
        """
        parts = [f"🤖 Agent Response for {plant_name}"]
        parts.append(f"⏰ {timestamp}")
        parts.append("")
        
        if response_text:
            parts.append(f"💭 Response:\n{response_text[:500]}")
            if len(response_text) > 500:
                parts.append("... (truncated)")
        
        if tool_results:
            parts.append("")
            parts.append(f"🔧 Executed {len(tool_results)} tool(s):")
            for tr in tool_results:
                tool_name = tr.get("tool", "unknown")
                result = tr.get("result", {})
                msg = result.get("message", result.get("success", "completed"))
                parts.append(f"  • {tool_name}: {msg}")
        
        return "\n".join(parts)
