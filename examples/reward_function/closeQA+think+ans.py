import re, warnings
from typing import Any, Dict, List, Tuple
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

warnings.filterwarnings("ignore", category=UserWarning)


PAIR_PATTERN   = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
                            re.DOTALL | re.I)
OPTION_PATTERN = re.compile(r"([A-G])\s*[)）\.]?", re.I) 


def _parse_pairs(text: str) -> List[Tuple[str, str]]:
    return PAIR_PATTERN.findall(text)


def _extract_option(ans_block: str) -> str:
    m = OPTION_PATTERN.search(ans_block.strip())
    return m.group(1).upper() if m else ""


def _find_think(pairs: List[Tuple[str, str]], target_opt: str) -> str:
    for th, ans in pairs:
        if _extract_option(ans) == target_opt:
            return th.strip()
    return ""



def format_reward(predict_str: str) -> float:
    pattern = r"(?:<think>.*?</think>\s*<answer>.*?</answer>\s*)+"
    ok = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if ok else 0.0



def _preprocess(text: str):
    return str(text).lower().replace(".", " .").split(" ")


def compute_metrics(pred: str, gt: str) -> Dict[str, float]:
    pred_tokens = _preprocess(pred)
    gt_tokens   = _preprocess(gt)

    # BLEU‑1
    bleu1 = sentence_bleu([gt_tokens], pred_tokens, weights=(1,))

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
    if not isinstance(reward_inputs, list):
        raise ValueError("not batch compute_score。")
    if not (0 <= format_weight <= 1):
        raise ValueError("format_weight not in 0~1")

    alpha = 0.3
    acc_weight = 1 - format_weight
    results = []

    for item in reward_inputs:
        resp = item["response"]
        gt   = item["ground_truth"]
        pairs_resp = _parse_pairs(resp)
        pairs_gt   = _parse_pairs(gt)

        fmt = format_reward(resp)

        pred_opt = _extract_option(_parse_pairs(resp)[-1][1] if _parse_pairs(resp) else "")
        true_opt = _extract_option(_parse_pairs(gt)[-1][1]   if _parse_pairs(gt)   else "")
        acc = 1.0 if pred_opt and pred_opt == true_opt else 0.0
        
        process = 0.0
        if pairs_resp and true_opt:
            all_correct = True
            for _, ans in pairs_resp:
                opt = _extract_option(ans)
                if ('✅' in ans and opt == true_opt) or ('❌' in ans and opt != true_opt):
                    continue
                all_correct = False
                break
            if all_correct:
                process = 0.5

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


        overall=format_weight * fmt + acc_weight * acc

        results.append(
            {
                "overall": round(overall, 6),
                "format":  fmt,
                "acc":     acc,
                "reason":  reason,
                "process": process,
                "base_reward": overall
            }
        )
    return results
