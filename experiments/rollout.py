import asyncio
from copy import deepcopy
import os
import traceback
from typing import List
import uuid
import json

import aiofiles
import aiofiles.os
import colorama
from experiments.config import Config
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.experimental.openai import ArealOpenAI
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

from mcp import ClientSession
from mcp_client import MCPClient
from sandbox import Sandbox
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger("Multi-Turn workflow")


def convert_tool_format(tool):
    input_schema = getattr(tool, "inputSchema", {}) or {}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "parameters": {
                "type": "object",
                "properties": input_schema.get("properties", {}),
                "required": input_schema.get("required", []),
            },
        },
    }


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=0, max=60),
)
async def call_llm(
    messages: list[dict[str, str]],
    tools: list[dict],
    api_client: ArealOpenAI,
):
    response = await api_client.chat.completions.create(
        messages=messages, tools=tools
    )
    return response

@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=0, max=60),
)
async def call_tool(mcp_session: ClientSession, tool_name: str, tool_args: dict):
    result = await mcp_session.call_tool(tool_name, tool_args)
    return result

def prepare_messages(data, prompt_key, system_prompt=None):
    messages = []
    if isinstance(system_prompt, str) and len(str) > 0:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": data[prompt_key]})
    return messages

def concat_sequence_dim(tensor_dicts: List[TensorDict]) -> TensorDict:
    """Concatenate tensors from multiple turns along sequence dimension."""
    if not tensor_dicts:
        return TensorDict()

    result = {}

    # Handle multi_modal_input separately
    if "multi_modal_input" in tensor_dicts[0]:
        # For multi-turn, typically use the first turn's multi_modal_input
        result["multi_modal_input"] = tensor_dicts[0]["multi_modal_input"]

    # Concatenate other tensors along sequence dimension
    for key in tensor_dicts[0].keys():
        if key == "multi_modal_input":
            continue

        tensors_to_concat = []
        for tensor_dict in tensor_dicts:
            tensor = tensor_dict[key]
            tensors_to_concat.append(tensor)

        # Concatenate along sequence dimension (dim=1) for 2D tensors
        if len(tensors_to_concat[0].shape) > 1:
            result[key] = torch.cat(tensors_to_concat, dim=1)
        else:
            # For 1D tensors (like rewards), take the last one or sum
            result[key] = tensors_to_concat[-1]  # Use last turn's value

    # Update batch size - keep first dimension, sum sequence lengths
    new_batch_size = [1]

    return TensorDict(result, batch_size=new_batch_size)

class AgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        config: Config,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.config = config
        self.max_turns = config.max_turns
        self.sandbox = Sandbox(base_url=config.sandbox_control_plane)
        self.sandbox_gateway = config.sandbox_gateway
        self.rollout_stat_scope = rollout_stat_scope
        # self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.async_reward_fn = reward_fn
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Create tokens that should be amended if the answer is incorrect.
        # This method eliminates the encode-decode inconsistency issue and cancels system prompts.
        messages = [{"role": "assistant", "content": "some random message."}]
        s1 = self.tokenizer.apply_chat_template(messages, tokenize=True)
        messages += [
            {
                "role": "user",
                "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                "Please carefully read the original question, check the preivous errors, and try to answer it again.",
            }
        ]
        s2 = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        self.multi_turn_prompt_ids = s2[len(s1) :]

    async def _run_one_episode(self, engine: InferenceEngine, data):
        sandbox_uuid = await self.sandbox.spawn(
            image=data["sandbox"]["image"] 
            if (image := self.config.sandbox_image_override) is None
            else image
        )
        await asyncio.sleep(10)
        logger.info(f"Spawned sandbox with UUID: {sandbox_uuid}")
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer, tool_call_parser='qwen25')

        mcp_client = MCPClient(self.sandbox_gateway)

        try:
            mcp = await mcp_client.attach_session(sandbox_uuid)
            tools = await mcp.list_tools()
            available_tools = [convert_tool_format(tool) for tool in tools.tools]
            messages = prepare_messages(data, "question")

            assert available_tools is not None, f"Tool cannot be None, {tools}"
            
            
            for _ in range(self.max_turns):
                response = await call_llm(
                    messages=messages,
                    tools=available_tools,
                    api_client=client,
                )
                if response.choices is None:
                    logger.error("LLM returned no choices.")
                    break
                messages.append(response.choices[0].message.model_dump())
                content = response.choices[0].message

                comp_data = client.get_completions(response.id)

                if content.tool_calls is None:
                    if content.content:
                        break
                    continue
                if len(content.tool_calls) == 0:
                    if content.content:
                        if '<tool_call>' in content.content:
                            messages += [
                                {
                                    "role": "user",
                                    "content": "Your tool call format or argument is incorrect or use a tool not provided to you, you need to carefully review the tools provided to you and regenerate your response."
                                }
                            ]
                            logger.info(f"Tool call is empty, reprompt to generate new response {content}")
                            continue
                        else:
                            break
                    else:
                        messages += [
                            {
                                "role": "user",
                                "content": "You don't give any answer, review the question and chat history and continue answer the question."
                            }
                        ]
                        continue


                logger.info(
                    f"Calling tool: {content.tool_calls[0].function.name}({content.tool_calls[0].function.arguments})"
                )

                tool_name = content.tool_calls[0].function.name
                tool_args = content.tool_calls[0].function.arguments
                try:
                    tool_args = json.loads(tool_args) if tool_args else {}
                    result = await call_tool(mcp, tool_name, tool_args)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": content.tool_calls[0].id,
                            "name": tool_name,
                            "content": [c.model_dump() for c in result.content],
                        }
                    )
                except json.JSONDecodeError:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": content.tool_calls[0].id,
                            "name": tool_name,
                            "content": "Error: Malformed tool arguments.",
                        }
                    )
                except Exception as e:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": content.tool_calls[0].id,
                            "name": tool_name,
                            "content": f"Error: {str(e)}",
                        }
                    )
                    
            try:
                reward = self.async_reward_fn(
                    example=data,
                    messages=messages,
                    mcp=mcp,
                    sandbox_uuid=sandbox_uuid,
                )
                client.set_reward(response.id, reward)
            except Exception:
                logger.error(
                    f"Reward function failed for problem {data['qid']}: {traceback.format_exc()}"
                )
                reward = {"score": 0, "metadata": {"error": "Reward function failed."}}
                with open("reward_errors.log", "a") as f:
                    f.write(f"{data['qid']}\n{traceback.format_exc()}\n\n")
        finally:
            await mcp_client.close_session()
            await self.sandbox.deprovision(sandbox_uuid)
        completions = client.export_completions()
        tensor_dicts = [completions[key].to_tensor_dict() for key in completions]
        res = concat_sequence_dim(tensor_dicts)

        return (
            TensorDict(res, batch_size=[1]),
            # prompt_str,
            # completions_str,
            reward,
            # len(seq),
        )


    async def arun_episode(self, engine: InferenceEngine, data):
        tasks = [
            self._run_one_episode(engine, data)
            for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)

        if self.dump_dir is not None:
            version = engine.get_version()
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                n_samples = self.gconfig.n_samples
                for i, (_, reward) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}",
                            f"reward: {reward}",
                            # f"messages:\n{colorama.Fore.YELLOW + colorama.Style.DIM}{json.dumps(completion.messages, indent=2)}{colorama.Style.RESET_ALL}",
                            # f"response:\n{colorama.Fore.YELLOW + colorama.Style.DIM}{json.dumps(completion.response, indent=2)}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        data = [res[0] for res in results]
        return concat_padded_tensors(data)
