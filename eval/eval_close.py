import os
import json
import base64
import openai
from PIL import Image
import io
import time
from tqdm import tqdm
import traceback

from openai import OpenAI

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info

# Load model and processor from a local checkpoint directory (replace with your path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "./checkpoints/qwen2_5_vl_7b_close+think+ans+open+think+ans_grpo/global_step_288/actor/huggingface",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(
    "./checkpoints/qwen2_5_vl_7b_close+think+ans+open+think+ans_grpo/global_step_288/actor/huggingface"
)

# Generation config (can be adjusted as needed)
temp_generation_config = GenerationConfig(
    max_new_tokens=2048,
    do_sample=False,
    temperature=1,
    num_return_sequences=1,
    pad_token_id=151643,
)


def call_gpt4o_api(prompt, images=None):
    """
    Build a multi-modal chat message and run generation with the local Qwen2.5-VL model.
    """
    if images is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        content = []
        for img_path in images:
            content.append({"type": "image", "image": img_path})
        content.append(
            {
                "type": "text",
                "text": (
                    prompt
                    + "\nPlease review and evaluate each option in sequence, "
                      "providing judgments for each, and summarize to output the final answer. "
                      "Please think step by step, and conduct your reasoning within <think></think> "
                      "and share intermediate answers within <answer></answer>. "
                      "Use an alternating reasoning format: "
                      "<think></think><answer></answer><think></think><answer></answer> "
                      "until you reach the final answer."
                ),
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

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

    # Generate
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        generation_config=temp_generation_config,
    )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def is_exact_match(pred, gold):
    """
    Exact string match after stripping and lower-casing.
    """
    return pred.strip().lower() == gold.strip().lower()


def evaluate_folder(folder_path, output_file):
    """
    Evaluate all .jsonl files under a folder.
    For each line (a sample), run the model, compare with gold answer, and write results.
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]

    # File-level progress bar
    for filename in tqdm(all_files, desc="Processing Files", total=len(all_files)):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            # Limit lines if needed; adjust or remove slicing
            lines = f.readlines()[:50]

        # Line-level progress bar
        for line in tqdm(lines, desc=f"Processing {filename}", leave=False):
            data = json.loads(line.strip())
            user_input = None
            gold_answer = None

            for msg in data.get("messages", []):
                if msg.get("role") == "user":
                    # Remove <image> placeholder if present
                    user_input = msg.get("content", "").replace("<image>", "").strip()
                elif msg.get("role") == "assistant":
                    gold_answer = msg.get("content", "").strip()

            image_paths = data.get("images", [])

            try:
                model_output = call_gpt4o_api(user_input, image_paths)
                correct = is_exact_match(model_output, gold_answer)

                # Attach prediction and correctness
                data["predict"] = model_output
                data["correct"] = 1 if correct else 0

                print(f"[Processed] ID: {data.get('id', 'unknown_id')} | gold_answer: {gold_answer} | model_output: {model_output}")

            except Exception as e:
                print(f"[Error] ID: {data.get('id', 'unknown_id')}")
                print("Exception:", str(e))
                data["predict"] = None
                data["correct"] = 0

            # Append updated record to output file
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Use relative paths or placeholders instead of absolute paths
    folder_path = "./data/stage2/test/disease_qa_m"  # e.g., PATH_TO_YOUR_TEST_JSONL_FOLDER
    output_file = "./outputs/exp/zcase_study/multi.jsonl"  # e.g., PATH_TO_OUTPUT_JSONL
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    evaluate_folder(folder_path, output_file)
