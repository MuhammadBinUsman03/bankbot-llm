"""
LLM client implementation using Hugging Face.
"""

from huggingface_hub import InferenceClient
import logging
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with LLM services."""
    
    def __init__(self, provider="novita", api_key=None, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize the LLM client.
        
        Args:
            provider (str): Provider name for the Hugging Face Inference API
            api_key (str): API key for authentication
            model_name (str): Name of the model to use
        """
        self.client = InferenceClient(
            provider=provider,
            api_key=api_key,
        )
        self.model_name = model_name
        logger.info(f"Initialized LLM client with model: {model_name}")
    
    def generate_answer(self, prompt_messages):
        """
        Generate answer using Hugging Face LLM.
        
        Args:
            prompt_messages: Messages in various formats
            
        Returns:
            str: Generated answer
        """
        messages = self._format_messages(prompt_messages)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Sorry, I'm having trouble processing your request."
    
    def _format_messages(self, prompt_messages) -> List[Dict[str, str]]:
        """
        Format messages to standard format expected by Hugging Face API.
        
        Args:
            prompt_messages: Messages in various formats
            
        Returns:
            List[Dict[str, str]]: Formatted messages
        """
        messages = []
        
        # Handle different message formats
        if isinstance(prompt_messages, list) and len(prompt_messages) > 0:
            # Case 1: List of messages with tuple format like ('messages', [HumanMessage(...)])
            if isinstance(prompt_messages[0], tuple) and prompt_messages[0][0] == 'messages':
                for message_obj in prompt_messages[0][1]:
                    if hasattr(message_obj, 'content'):
                        # Convert LangChain message to dict format
                        role = "assistant" if hasattr(message_obj, 'type') and message_obj.type == "ai" else "user"
                        messages.append({"role": role, "content": message_obj.content})
            # Case 2: Direct list of message objects
            elif hasattr(prompt_messages[0], 'content'):
                for msg in prompt_messages:
                    role = "assistant" if hasattr(msg, 'type') and msg.type == "ai" else "user"
                    messages.append({"role": role, "content": msg.content})
            # Case 3: Already formatted as dicts
            elif isinstance(prompt_messages[0], dict) and "role" in prompt_messages[0]:
                messages = prompt_messages
        # Case 4: Single string message - wrap as user message
        elif isinstance(prompt_messages, str):
            messages = [{"role": "user", "content": prompt_messages}]
        
        # If no messages were successfully parsed, create a default
        if not messages:
            logger.warning("Failed to parse message format. Using empty prompt.")
            messages = [{"role": "user", "content": "Please provide information."}]
            
        return messages

