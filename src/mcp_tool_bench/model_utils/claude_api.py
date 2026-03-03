# #!/usr/bin/python
# # -*- coding: UTF-8 -*-
import json
import logging
import requests
from typing import List, Dict, Any, Optional
import os
import sys
import anthropic

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '../')))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, './')))

from src.mcp_tool_bench.model_utils.base_api import *
from src.mcp_tool_bench.global_variables import settings

class ClaudeModelAPIProvider(BaseModelAPIProvider):
    """
    Anthropic Claude API for chat and tool use.
    https://docs.anthropic.com/en/docs/tool-use
    """
    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def api_chat(self, messages: List, **kwargs) -> Dict[str, Any]:
        """
        Claude chat completion.
        """
        try:
            model = self.model_name
            if not model:
                model = MODEL_SELECTION_CLAUDE_37

            system_message_content = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message_content += msg["content"] + "\n"
                else:
                    chat_messages.append(msg)

            create_kwargs = {
                "model": model,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "messages": chat_messages,
                "temperature": kwargs.get("temperature", 0.3),
            }
            if system_message_content.strip():
                create_kwargs["system"] = [{"type": "text", "text": system_message_content.strip()}]
            response = self.client.messages.create(**create_kwargs)
            completion, reasoningContent = post_process_claude_chat_response(response)
            result = {
                KEY_FUNCTION_CALL: {},
                KEY_COMPLETION: completion,
                KEY_REASON_CONTENT: reasoningContent
            }
            return result

        except Exception as e:
            logging.error(f"Failed to process Claude api_chat: {e}")
            return {}

    def api_function_call(self, messages: List, tools: List, **kwargs) -> Dict[str, Any]:
        """
        Claude tool use (function calling).
        Args:
            messages: List of message [{}, {}]
            tools: List of tool definitions in Claude's format (e.g., from tools_claude_wrapper)
        """
        try:
            model = self.model_name
            if not model:
                model = MODEL_SELECTION_CLAUDE_37

            system_message_content = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message_content += msg["content"] + "\n"
                else:
                    chat_messages.append(msg)

            # Convert OpenAI-format tools to Claude format if needed
            claude_tools = []
            for tool in tools:
                if "type" in tool and tool["type"] == "function" and "function" in tool:
                    # OpenAI format -> Claude format
                    func = tool["function"]
                    claude_tools.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {})
                    })
                else:
                    claude_tools.append(tool)

            create_kwargs = {
                "model": model,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "messages": chat_messages,
                "tools": claude_tools,
                "tool_choice": kwargs.get("tool_choice", {"type": "auto"}),
                "temperature": kwargs.get("temperature", 0.3),
            }
            if system_message_content.strip():
                create_kwargs["system"] = [{"type": "text", "text": system_message_content.strip()}]
            response = self.client.messages.create(**create_kwargs)
            tool_result, completion, reasoningContent = post_process_claude_function_call_response(response)
            result = {
                KEY_FUNCTION_CALL: tool_result,
                KEY_COMPLETION: completion,
                KEY_REASON_CONTENT: reasoningContent
            }
            return result

        except Exception as e:
            logging.error(f"Failed to process Claude api_function_call: {e}")
            return {}

def post_process_claude_chat_response(response: Any) -> (str, str):
    """
    Processes the response from Claude chat completion.
    Claude's response content is a list of content blocks.
    """
    if response is None or not response.content:
        return "", ""

    completion_content = ""
    reasoning_content = "" # Claude might have thinking blocks

    for block in response.content:
        if block.type == "text":
            completion_content += block.text
    return completion_content, reasoning_content

def post_process_claude_function_call_response(response: Any) -> (Dict[str, Any], str, str):
    """
    Processes the response from Claude for tool use.
    Extracts the tool call details and any text response.
    """
    if response is None or not response.content:
        return {}, "", ""
    tool_call_result = {}
    completion_content = ""
    reasoning_content = ""

    try:

        for block in response.content:
            if block.type == "tool_use":
                if not tool_call_result: 
                    tool_call_result = {
                        "function_name": block.name,
                        "function_arguments": block.input, # Claude's tool_use.input is already a dict
                        "is_function_call": True,
                        "id": block.id # Store tool_use ID for sending tool results back
                    }
            elif block.type == "text":
                completion_content += block.text
        return tool_call_result, completion_content, reasoning_content
            
    except Exception as e:
        print (f"DEBUG: Failed to post_process_claude_function_call_response with error {e}")       
        return tool_call_result, completion_content, reasoning_content

if __name__ == '__main__':    
    # Test function calling
    user_prompt = "Weather query template"
    system_prompt = ""
    try:
        messages = [{"role": "user", "content": user_prompt}]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        input_file = os.path.join(package_dir, "mcp/tools/demo/demo_tools.json")
        tools = json.load(open(input_file, "r", encoding="utf-8"))

        api_provider = ClaudeModelAPIProvider(MODEL_SELECTION_CLAUDE_37)
        result = api_provider.api_function_call(messages, tools)
        print("Function Call Response:", result)
    except FileNotFoundError:
        print("Demo tools file not found, skipping function call test")
