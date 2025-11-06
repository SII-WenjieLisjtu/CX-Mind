import os
import json
import base64
import re
from collections import Counter
from tqdm import tqdm
import traceback

from openai import AzureOpenAI

client2 = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://XXXXX.azure-api.net"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "YOUR_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
)

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info

# ----- Load local model & processor (replace with your relative path or placeholder) -----
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "./checkpoints/qwen2_5_vl_7b_close+think+ans+open+think+ans_grpo/global_step_288/actor/huggingface",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(
    "./checkpoints/qwen2_5_vl_7b_close+think+ans+open+think+ans_grpo/global_step_288/actor/huggingface"
)

# ----- Text generation configuration -----
temp_generation_config = GenerationConfig(
    max_new_tokens=2048,
    do_sample=True,
    temperature=1,
    num_return_sequences=1,
    pad_token_id=151643,
)


def call_gpt4o_api(prompt, images=None):
    """
    Build multi-modal chat messages and run local Qwen2.5-VL generation.
    'images' should be a list of image paths (strings).
    """
    if images is None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    else:
        content = []
        for img_path in images:
            content.append({"type": "image", "image": img_path})
        content.append({
            "type": "text",
            "text": (
                prompt
                + "\nPlease first provide potential diagnoses, then evaluate each one in sequence, "
                  "and summarize to output the final answer. Please think step by step, and conduct "
                  "your reasoning within <think></think> and share intermediate answers within "
                  "<answer></answer>. Use an alternating reasoning format: "
                  "<think></think><answer></answer><think></think><answer></answer> "
                  "until you reach the final answer."
            ),
        })
        messages = [{"role": "user", "content": content}]

    # Build model inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Run generation (rely on GenerationConfig to avoid conflicting args)
    generated_ids = model.generate(**inputs, generation_config=temp_generation_config)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def call_gpt4o_api2(prompt, images=None):
    """
    Call Azure OpenAI (GPT-4o) for equivalence judgment.
    'images' should be a list of base64-encoded strings if provided.
    """
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": prompt}] +
                (
                    [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                        for img in images if img is not None
                    ] if images else []
                )
            ),
        }
    ]

    try:
        response = client2.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] GPT API call failed: {e}")
        return ""


def calculate_jaccard(text1, text2):
    """
    Compute Jaccard similarity between two strings after tokenization.
    """
    def tokenize(text):
        return set(re.findall(r"\w+", text.lower()))

    tokens1 = tokenize(text1 or "")
    tokens2 = tokenize(text2 or "")
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0


def evaluate_with_gpt(pred, gold):
    """
    Use GPT to judge whether prediction and gold answer are semantically equivalent.
    Falls back to Jaccard-only if parsing fails.
    """
    if not pred or not gold:
        return {"correct": 0, "jaccard": 0.0}

    prompt = (
        "You are a medical QA evaluator. "
        "Given the correct answer and a model prediction, determine if they mean the same thing.\n\n"
        f"Correct Answer: {gold}\n"
        f"Model Prediction: {pred}\n\n"
        "Respond strictly in JSON format:\n"
        "{\"correct\": 0 or 1, \"jaccard\": calculated_jaccard_score}"
    )

    try:
        gpt_response = call_gpt4o_api2(prompt)
        result = json.loads(gpt_response.replace("```json", "").replace("```", ""))
        print(result)
        return result
    except Exception:
        print("[Warning] Could not parse GPT evaluation. Falling back to Jaccard only.")
        jaccard = calculate_jaccard(pred, gold)
        return {"correct": 1 if jaccard > 0.7 else 0, "jaccard": jaccard}


def evaluate_folder(folder_path, output_file):
    """
    Evaluate all .jsonl files under a folder.
    For each line: build inputs, run local model, evaluate with GPT (or Jaccard),
    and append results to 'output_file'.
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]

    for filename in tqdm(all_files, desc="Processing Files", total=len(all_files)):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:50]  # adjust or remove this slice for full evaluation

        for line in tqdm(lines, desc=f"Processing {filename}", leave=False):
            data = {}
            try:
                data = json.loads(line.strip())
                user_input = None
                gold_answer = None

                for msg in data.get("messages", []):
                    if msg.get("role") == "user":
                        user_input = msg.get("content", "").replace("<image>", "").strip()
                    elif msg.get("role") == "assistant":
                        gold_answer = msg.get("content", "").strip()

                image_paths = data.get("images", [])

                model_output = call_gpt4o_api(user_input, image_paths)
                print(f"model output: {model_output} | gold_answer: {gold_answer}")
                eval_result = evaluate_with_gpt(model_output, gold_answer)

                correct = int(eval_result.get("correct", 0))
                jaccard = float(eval_result.get("jaccard", calculate_jaccard(model_output, gold_answer)))

                data["predict"] = model_output
                data["correct"] = correct
                data["jaccard"] = jaccard

                print(f"[Processed] ID: {data.get('id', 'unknown')} | Correct: {correct} | Jaccard: {jaccard:.3f}")

            except Exception:
                print(f"[Error] Processing line:\n{traceback.format_exc()}")
                # Record a safe fallback entry on error
                data.setdefault("predict", None)
                data.setdefault("correct", 0)
                data.setdefault("jaccard", 0.0)

            # Append the (possibly updated) record to output file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Use relative paths or placeholders rather than absolute paths
    folder_path = "./data/stage2/test/disease_open"          # e.g., PATH_TO_YOUR_JSONL_FOLDER
    output_file = "./outputs/exp/zcase_study/open.jsonl"     # e.g., PATH_TO_OUTPUT_JSONL
    evaluate_folder(folder_path, output_file)
