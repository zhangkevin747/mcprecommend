# #!/usr/bin/python
# # -*- coding: UTF-8 -*-
import json
import logging
import requests
from typing import List, Dict, Any, Optional
import os
import sys
import openai
from openai import OpenAI

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '../')))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, './')))

from src.mcp_tool_bench.model_utils.base_api import *
from src.mcp_tool_bench.global_variables import settings

def tools_openai_wrapper(tools):
    tools_wrapped = [{
        "type": "function",
        "function":{
            "name": tool["name"] if "name" in tool else "", 
            "description": tool["description"] if "description" in tool else "",
            "parameters": tool["input_schema"] if "input_schema" in tool else {}
        }
    } for tool in tools]
    return tools_wrapped

class OpenAIModelAPIProvider(BaseModelAPIProvider):
    """
    OpenAI API for chat and function calling.
    https://platform.openai.com/docs/api-reference/chat
    """
    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.openai.com/v1"
        )

    def api_chat(self, messages: List, **kwargs) -> Dict[str, Any]:
        """
        OpenAI chat completion.
        """
        try:
            model = self.model_name
            if not model: 
                model = "gpt-4o"

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.3)
            )
            completion, reasoningContent = post_process_openai_chat_response(response)
            result = {
                KEY_FUNCTION_CALL: {},
                KEY_COMPLETION: completion,
                KEY_REASON_CONTENT: reasoningContent
            }
            return result

        except Exception as e:
            logging.error(f"Failed to process OpenAI api_chat: {e}")
            return {}

    def api_function_call(self, messages: List, tools: List, **kwargs) -> Dict[str, Any]:
        """
        OpenAI function calling (tool calling).
        Args:
            messages: List of message [{}, {}]
            tools: List of tool definitions [{type: "function", function: {name: "", description: "", parameters: {}}}]
        """
        try:
            model = self.model_name
            if not model:
                model = "gpt-4o" 
            create_kwargs = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            # GPT-5 only supports default temperature (1)
            if not model.startswith("gpt-5"):
                create_kwargs["temperature"] = kwargs.get("temperature", 0.3)
            response = self.client.chat.completions.create(**create_kwargs)
            tool_result = post_process_openai_function_call_response(response)
            tool_call_mapped, completion, reasoningContent = function_call_result_common_mapper(tool_result)

            result = {
                KEY_FUNCTION_CALL: tool_call_mapped,
                KEY_COMPLETION: "",
                KEY_REASON_CONTENT: ""
            }
            return result

        except Exception as e:
            logging.error(f"Failed to process OpenAI api_function_call: {e}")
            return {}

def post_process_openai_chat_response(response):
    """
    Processes the response from OpenAI chat completion.
    """
    if response is None or not response.choices:
        return "", ""
    completion_content = ""
    if response.choices[0].message.content:
        completion_content = response.choices[0].message.content
    return completion_content, ""

def post_process_openai_function_call_response(response):
    """
    Processes the response from OpenAI for function calls.
    Extracts the tool call details.
    """
    if response is None or not response.choices or not response.choices[0].message:
        return {}

    try:
        message = response.choices[0].message
        if message.tool_calls:
            first_tool_call = message.tool_calls[0]
            if first_tool_call.type == "function" and first_tool_call.function:
                tool_call = {
                    "id": first_tool_call.id,
                    "function": {
                        "name": first_tool_call.function.name,
                        "arguments": first_tool_call.function.arguments
                    }
                }
                return tool_call
        return {}
    except Exception as e:
        print (f"Failed to post_process_openai_function_call_response error {e}")
        return {}

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
        wrappered_tools = tools_openai_wrapper(tools)

        gpt_api_provider = OpenAIModelAPIProvider()
        result = gpt_api_provider.api_function_call(messages, wrappered_tools)
        print("Function Call Response:", result)
    except FileNotFoundError:
        print("Demo tools file not found, skipping function call test")
