import asyncio
import json
import re
from typing import Any

from config import Config
from mcp import ClientSession
from reasoning_gym import get_score_answer_fn
from sandbox_fusion import compute_score

from math_eval import math_equal
import numpy as np

def quick_parse(text):
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text

def equal_func(answer, ground_truth):
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def last_boxed_only_string(string):
    if type(string) != str:
        string = str(string)
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def extract_number(s: str) -> float:
    """Extract the first number found in the string"""
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if match:
        return float(match.group(0))
    else:
        return None

async def reward_fn(
    example: dict,
    messages: list[dict[str, str]],
    mcp: ClientSession,
    sandbox_uuid: str,
    config: Config,
) -> dict[str, Any]:
    extra = json.loads(example["extra"])

    last_message_content = (
        messages[-1]["content"] if messages[-1]["content"] is not None else ""
    )
    try:
        if example["source"] == "zwhe99/DeepMath-103K":
            answer = extract_boxed_answer(last_message_content)
            if answer is None:
                return {"score": 0, "metadata": {"parsed_answer": None, "last_content": last_message_content}}
            return {
                "score": 1 if equal_func(extra['final_answer'], answer) else 0,
                "metadata": {"parsed_answer": answer},
            }

        elif example["source"] == "inclusionAI/ASearcher-train-data":
            answer = extract_boxed_answer(last_message_content)
            if answer is None:
                return {"score": 0, "metadata": {"parsed_answer": None}}
            answer = answer.lower()
            for option in extra["answer"] + extra["aug_answer"]:
                if option is not None and option.lower() in answer:
                    return {"score": 1, "metadata": {"parsed_answer": answer}}
            return {"score": 0, "metadata": {"parsed_answer": answer}}

        elif example["source"] == "open-thought/reasoning-gym":
            answer = extract_boxed_answer(last_message_content)
            if answer is None:
                return {"score": 0, "metadata": {"parsed_answer": None}}
            extra["metadata"] = json.loads(extra["metadata"])
            rg_score_fn = get_score_answer_fn(extra["metadata"]["source_dataset"])
            score = rg_score_fn(answer, extra)
            return {"score": score, "metadata": {"parsed_answer": answer}}

        elif example["source"] == "BAAI/TACO":
            answer = last_message_content
            if answer is None:
                return {"score": 0.0, "metadata": {"parsed_answer": None}}
            test_cases = json.loads(extra["input_output"])
            memory_limit_mb = 512
            if "memory_limit" in extra and extra["memory_limit"] is not None:
                memory_limit_mb = extract_number(extra["memory_limit"]) or 512
            timeout = 5
            if "timeout" in extra and extra["timeout"] is not None:
                timeout = extract_number(extra["timeout"]) or 5
            score_metadata = await asyncio.to_thread(
                lambda: compute_score(
                    sandbox_fusion_url=config.sandbox_fusion_url,
                    concurrent_semaphore=None,
                    memory_limit_mb=memory_limit_mb,
                    timeout=int(timeout),
                    completion=answer,
                    test_cases=test_cases,
                    continuous=True,
                )
            )

            if score_metadata is None:
                return {"score": 0.0, "metadata": {"parsed_answer": None}}

            score, final_metadata = score_metadata
            return {"score": score, "metadata": final_metadata}

        else:
            raise ValueError(f"Unknown dataset source: {example['source']}")
    except Exception as e:
        print(f"Error in reward function: {e}")
        return {"score": 0.0, "metadata": {"error": str(e), "last_content": last_message_content}}