import re
from typing import Any, Dict, List

# ---------- 严格交替格式评分 ----------
def format_reward(response: str) -> float:
    """
    满足：
      1) 整篇由若干 <think>...</think><answer>...</answer> 串联
      2) 标签数量一一对应
    则得 1.0；否则 0.0
    """
    clean = re.sub(r"\s*(<|>|/)\s*", r"\1", response.strip())
    strict_pattern = re.compile(
        r'^(?:<think>.*?</think>\s*<answer>.*?</answer>\s*)+$',
        re.DOTALL
    )
    if not re.fullmatch(strict_pattern, clean):
        return 0.0
    think_cnt = len(re.findall(r"<think>", clean))
    ans_cnt   = len(re.findall(r"<answer>", clean))
    return 1.0 if think_cnt == ans_cnt else 0.0


# ---------- 提取最终 <answer> ----------
def extract_final_answer(response: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return matches[-1] if matches else ""


# ---------- 提取选项字母 ----------
OPTION_PATTERN = re.compile(r"([A-D])\s*[)）\.]?", re.I)

def extract_option(text: str) -> str:
    """
    支持格式：A)、A), A.、A ）等，大小写皆可。
    只返回首个匹配的大写字母；若无则返回空串。
    """
    m = OPTION_PATTERN.search(text.strip())
    return m.group(1).upper() if m else ""


# ---------- 评分主函数 ----------
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("请以列表批量调用 compute_score。")

    results = []
    for item in reward_inputs:
        response      = item["response"]
        ground_truth  = item["ground_truth"]

        # 1) 格式分
        fmt = format_reward(response)

        # 2) 预测选项（来自最终 <answer>）
        pred_block   = extract_final_answer(response)
        pred_option  = extract_option(pred_block)

        # 3) 真实选项（同样从 ground_truth 的最终 <answer> 中提取）
        true_block   = extract_final_answer(ground_truth)
        true_option  = extract_option(true_block)

        # 4) 命中得分
        acc = 1.0 if pred_option and (pred_option == true_option) else 0.0

        results.append(
            {
                "overall": format_weight * fmt + (1 - format_weight) * acc,
                "format" : fmt,
                "acc"    : acc,
            }
        )
    return results
