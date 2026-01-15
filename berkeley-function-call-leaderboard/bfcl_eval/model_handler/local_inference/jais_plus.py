from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class JaisPlusHandler(OSSHandler):
    """
    Handler for JaisPlus model with custom Llama-style chat template.

    Key differences from standard Llama:
    - Uses 'ai' role instead of 'assistant'
    - Tools listed as JSON objects within <tools> tags
    - Tool calls use <tool_call>{"name": "...", "arguments": {...}}</tool_call> format
    - Uses 'tool_response' role for execution results with <tool_call_response> tags
    - Has optional <think> tags for reasoning

    Chat template format:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    System prompt
    <tools>
    {"name": "func1", "description": "...", "parameters": {...}}
    {"name": "func2", "description": "...", "parameters": {...}}
    </tools>
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    User query
    <|eot_id|>
    <|start_header_id|>ai<|end_header_id|>
    <think>...</think>
    <tool_call>{"name": "func1", "arguments": {...}}</tool_call>
    <|eot_id|>
    <|start_header_id|>tool_response<|end_header_id|>
    <tool_call_response>result</tool_call_response>
    <|eot_id|>
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    @override
    def _format_prompt(self, messages, function):
        """
        Format messages into the custom Llama-style template.
        Maps 'assistant' role to 'ai' and 'tool' role to 'tool_response'.
        """
        formatted_prompt = "<|begin_of_text|>"

        for message in messages:
            role = message["role"]
            content = message["content"]

            # Map roles to custom template roles
            if role == "assistant":
                role = "ai"
            elif role == "tool":
                role = "tool_response"

            formatted_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content.strip()}<|eot_id|>"

        # Add generation prompt for ai response
        formatted_prompt += "<|start_header_id|>ai<|end_header_id|>\n\n"

        return formatted_prompt

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parse the model response, handling optional <think> tags.
        Extracts reasoning content if present.
        """
        model_response = api_response.choices[0].text

        reasoning_content = ""
        cleaned_response = model_response

        # Extract reasoning content from <think> tags if present
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            if "<think>" in parts[0]:
                reasoning_content = parts[0].split("<think>")[-1].strip()
            cleaned_response = parts[-1].strip()

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant (ai) message to the conversation, preserving reasoning content.
        """
        inference_data["message"].append(
            {
                "role": "assistant",  # Will be mapped to 'ai' in _format_prompt
                "content": model_response_data["model_responses"],
                "reasoning_content": model_response_data.get("reasoning_content", ""),
            }
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add execution results using the 'tool' role (mapped to 'tool_response' in _format_prompt).
        Wraps each result in <tool_call_response> tags.
        """
        for execution_result in execution_results:
            inference_data["message"].append(
                {
                    "role": "tool",  # Will be mapped to 'tool_response' in _format_prompt
                    "content": f"<tool_call_response>{execution_result}</tool_call_response>",
                }
            )

        return inference_data
