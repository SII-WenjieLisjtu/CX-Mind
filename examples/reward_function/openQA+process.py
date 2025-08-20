import re
from typing import Any, Dict, List
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

# ---------- 格式评分：严格交替 ----------
def format_reward(response: str) -> float:
    """
    只有满足下列全部条件才返回 1.0：
      1. 去除多余空白后，正文完全由若干 <think>...</think><answer>...</answer> 片段串联而成
      2. 标签数量配对一致
    否则返回 0.0
    """
    # 统一去掉标签内部多余空格，如 "< / think >"
    clean = re.sub(r"\s*(<|>|/)\s*", r"\1", response.strip())

    # 正则：多个交替片段，必须以 <think> 起始、以 </answer> 结束
    strict_pattern = re.compile(
        r'^(?:<think>.*?</think>\s*<answer>.*?</answer>\s*)+$',
        re.DOTALL
    )
    if not re.fullmatch(strict_pattern, clean):
        return 0.0

    # 进一步确保四类标签数目一致
    counts = {
        t: len(re.findall(fr"<{t}>", clean))
        for t in ["think", "answer"]
    }
    if counts["think"] == counts["answer"]:
        return 1.0
    return 0.0


# ---------- 提取最终 <answer> ----------
def extract_final_answer(response: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return matches[-1] if matches else ""


# ---------- 简单切分疾病名称 ----------
def extract_diseases(text: str) -> List[str]:
    # 按逗号 / 分号 / 换行分割，全部转小写去空格
    return [p.strip().lower() for p in re.split(r"[,;\n]", text) if p.strip()]


# openQA.py

def f1_reward(pred, truth):
    # 1) 特殊情况
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0

    pred_set  = {p.lower().strip() for p in pred}
    truth_set = {t.lower().strip() for t in truth}

    mlb = MultiLabelBinarizer()
    mlb.fit([pred_set | truth_set])            
    y_true = mlb.transform([truth_set])
    y_pred = mlb.transform([pred_set])

    return f1_score(y_true, y_pred,
                    average="micro", zero_division=0)



# ---------- 评分主函数 ----------
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("请以列表批量调用 compute_score。")

    results = []
    for item in reward_inputs:
        response = item["response"]

        # 1) 格式分
        fmt = format_reward(response)

        # 2) 预测疾病（来自最终 <answer>）
        pred_block = extract_final_answer(response)
        pred_diseases = extract_diseases(pred_block)

        # 3) 真实疾病：示例中直接由 item["answer"] 转 List[str]
        true_response = item["ground_truth"]
        true_block=extract_final_answer(true_response)
        true_diseases=extract_diseases(true_block)

        f1 = f1_reward(pred_diseases, true_diseases)

        results.append(
            {
                "overall": format_weight * fmt + (1 - format_weight) * f1,
                "format": fmt,
                "f1": f1,
            }
        )
    return results
