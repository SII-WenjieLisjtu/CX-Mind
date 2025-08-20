"""
文件路径：your_project/reward_functions/accuracy_reward.py

说明
----
- 评分结构：
    overall = format_weight * format_score           # 默认 0.1
             + (1 - format_weight) * acc_score       # 默认 0.9
             + reason_score                          # 直接累加

- reason_score 仅在 **预测选项命中** 时计算，
  = (BLEU‑1 + ROUGE‑L) / 2，计算方式沿用题主给出的代码。

依赖
----
pip install rouge-score nltk
"""

import re, warnings
from typing import Any, Dict, List, Tuple
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- 工具函数 -----------------------------
PAIR_PATTERN   = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
                            re.DOTALL | re.I)
OPTION_PATTERN = re.compile(r"([A-G])\s*[)）\.]?", re.I)   # 视任务范围调整


def _parse_pairs(text: str) -> List[Tuple[str, str]]:
    """提取顺序对应的 (think, answer) 列表"""
    return PAIR_PATTERN.findall(text)


def _extract_option(ans_block: str) -> str:
    """从 answer 片段中提取选项字母"""
    m = OPTION_PATTERN.search(ans_block.strip())
    return m.group(1).upper() if m else ""


def _find_think(pairs: List[Tuple[str, str]], target_opt: str) -> str:
    """在 pairs 中找到 answer 选项 == target_opt 的 think"""
    for th, ans in pairs:
        if _extract_option(ans) == target_opt:
            return th.strip()
    return ""


# ----------------------------- 格式评分 -----------------------------
def format_reward(predict_str: str) -> float:
    """
    满足全文严格由 <think><answer> 交替组成则返回 1，否则 0。
    """
    pattern = r"(?:<think>.*?</think>\s*<answer>.*?</answer>\s*)+"
    ok = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if ok else 0.0


# ----------------------------- BLEU & ROUGE -----------------------------
def _preprocess(text: str):
    """小写 + 句点分词，与示例保持一致"""
    return str(text).lower().replace(".", " .").split(" ")


def compute_metrics(pred: str, gt: str) -> Dict[str, float]:
    """
    返回 BLEU‑1, ROUGE‑1, ROUGE‑L（四位小数）
    """
    pred_tokens = _preprocess(pred)
    gt_tokens   = _preprocess(gt)

    # BLEU‑1
    bleu1 = sentence_bleu([gt_tokens], pred_tokens, weights=(1, 0, 0, 0))

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gt, pred)
    rouge1_f = scores["rouge1"].fmeasure
    rougeL_f = scores["rougeL"].fmeasure

    return {
        "BLEU-1": round(bleu1, 4),
        "ROUGE-1": round(rouge1_f, 4),
        "ROUGE-L": round(rougeL_f, 4),
    }


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
) -> List[Dict[str, float]]:
    """
    输入示例：
        reward_inputs = [
            {
                "response":      "<think>...</think><answer>...</answer> ...",
                "ground_truth":  "<think>...</think><answer>...</answer> ..."
            },
            ...
        ]
    返回：
        [{"overall":..., "format":..., "acc":..., "reason":...}, ...]
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("请以列表批量调用 compute_score。")
    if not (0 <= format_weight <= 1):
        raise ValueError("format_weight 必须在 0~1 之间")

    alpha = 0.3
    acc_weight = 1 - format_weight
    results = []

    for item in reward_inputs:
        resp = item["response"]
        gt   = item["ground_truth"]
        # pairs_resp = _parse_pairs(resp)
        # pairs_gt   = _parse_pairs(gt)

        #  格式分
        fmt = format_reward(resp)

        #  选项准确分 (封闭题)
        pred_opt = _extract_option(_parse_pairs(resp)[-1][1] if _parse_pairs(resp) else "")
        true_opt = _extract_option(_parse_pairs(gt)[-1][1]   if _parse_pairs(gt)   else "")
        acc = 1.0 if pred_opt and pred_opt == true_opt else 0.0
        
        # #  过程分：每轮判断对真实标签(True positive/negative)
        # process = 0.0
        # if pairs_resp and true_opt:
        #     all_correct = True
        #     for _, ans in pairs_resp:
        #         opt = _extract_option(ans)
        #         # 如果标记为✅，则选项必须是真正正确的；标记为❌，则选项必须不是真正正确的
        #         if ('✅' in ans and opt == true_opt) or ('❌' in ans and opt != true_opt):
        #             continue
        #         all_correct = False
        #         break
        #     if all_correct:
        #         process = 0.5

        #  reason 分 (仅 acc==1 时计算)
        reason = 0.0
        if  acc == 1.0:
            pred_think = _find_think(_parse_pairs(resp), true_opt)
            gt_think   = _find_think(_parse_pairs(gt),   true_opt)
            if pred_think and gt_think:
                m = compute_metrics(pred_think, gt_think)
                # print("----------------------")
                # print("pred:")
                # print(pred_think)
                # print("gt:")
                # print(gt_think)
                # print("----------------------")
                reason = alpha*float(m["BLEU-1"]) +  (1-alpha) * float(m["ROUGE-L"])


        # overall = format_weight * fmt + acc_weight * acc + reason
        overall=format_weight * fmt + acc_weight * acc

        results.append(
            {
                "overall": round(overall, 6),
                "format":  fmt,
                "acc":     acc,
                "reason":  reason,
                # "process": process
                "base_reward": overall
            }
        )
    return results
