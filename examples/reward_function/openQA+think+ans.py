import re
from typing import Any, Dict, List, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# ----------------------------- 工具函数 -----------------------------
PAIR_PATTERN = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL | re.I)

def _parse_pairs(text: str) -> List[Tuple[str, str]]:
    """提取所有 (think, answer) 对列表"""
    return PAIR_PATTERN.findall(text)


def extract_final_answer(response: str) -> str:
    """提取最后一轮 <answer> 内容"""
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return matches[-1] if matches else ""


def extract_diseases(text: str) -> List[str]:
    """按逗号/分号/换行分割，转小写去空格"""
    return [p.strip().lower() for p in re.split(r"[,;\n]", text) if p.strip()]


def f1_reward(pred: List[str], truth: List[str]) -> float:
    """多标签 F1 分数"""
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    pred_set = {p.lower().strip() for p in pred}
    truth_set = {t.lower().strip() for t in truth}
    mlb = MultiLabelBinarizer()
    mlb.fit([pred_set | truth_set])
    y_true = mlb.transform([truth_set])
    y_pred = mlb.transform([pred_set])
    return f1_score(y_true, y_pred, average="micro", zero_division=0)


def compute_metrics(pred: str, gt: str) -> Dict[str, float]:
    """计算 BLEU-1 和 ROUGE-L"""
    pred_tokens = pred.lower().replace('.', ' .').split()
    gt_tokens   = gt.lower().replace('.', ' .').split()
    bleu1 = sentence_bleu([gt_tokens], pred_tokens, weights=(1,))
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL_f = scorer.score(gt, pred)['rougeL'].fmeasure
    return {"BLEU-1": bleu1, "ROUGE-L": rougeL_f}

# ---------- 格式评分 ----------
def format_reward(response: str) -> float:
    clean = re.sub(r"\s*(<|>|/)\s*", r"\1", response.strip())
    strict = re.compile(r'^(?:<think>.*?</think>\s*<answer>.*?</answer>\s*)+$', re.DOTALL)
    return 1.0 if strict.fullmatch(clean) else 0.0

# ---------- 评分主函数 ----------
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    process_weight: float = 0.2,
    alpha: float = 0.3
) -> List[Dict[str, float]]:
    results = []
    for item in reward_inputs:
        resp = item.get("response", "")
        gt   = item.get("ground_truth", "")

        # 1) 格式分
        fmt = format_reward(resp)

        # 2) 提取最终疾病列表 F1
        pred_block = extract_final_answer(resp)
        true_block = extract_final_answer(gt)
        pred_diseases = extract_diseases(pred_block)
        true_diseases = extract_diseases(true_block)
        f1 = f1_reward(pred_diseases, true_diseases)

        # 3) 解析所有 think-answer 对
        pairs_pred = _parse_pairs(resp)
        pairs_gt   = _parse_pairs(gt)

        # 4) process 分：对中间每轮✅/❌判断是否正确
        process = 0.0
        if len(pairs_pred) > 2:
            # 跳过第 0 轮和最后一轮
            all_ok = True
            for th, ans in pairs_pred[1:-1]:
                disease = re.sub(r"[✅❌]", "", ans).strip().lower()
                if ('✅' in ans and disease in true_diseases) or \
                   ('❌' in ans and disease not in true_diseases):
                    continue
                all_ok = False
                break
            if all_ok:
                process = process_weight

        # 5) reason 分：对所有预测命中且标✅的轮次，计算推理指标并取均值
        reason_scores: List[float] = []
        for th, ans in pairs_pred[1:-1]:
            disease = re.sub(r"[✅❌]", "", ans).strip().lower()
            if '✅' in ans and disease in true_diseases:
                # 对应 gt 的 think
                gt_think = ''
                for th_g, ans_g in pairs_gt:
                    if disease in [d.strip().lower() for d in extract_diseases(ans_g)]:
                        gt_think = th_g.strip()
                        break
                if gt_think:
                    m = compute_metrics(th.strip(), gt_think)
                    reason_scores.append(alpha * m['BLEU-1'] + (1 - alpha) * m['ROUGE-L'])
        reason = sum(reason_scores) / len(reason_scores) if reason_scores else 0.0

        # 6) overall
        overall = format_weight * fmt + (1 - format_weight) * f1 

        results.append({
            "overall": round(overall, 6),
            "format": fmt,
            "f1": f1,
            "reason": reason,
            "process": process,
            "base_reward": overall,
        })

    return results
