import transformers
import torch
import numpy as np
import random
import os
import json
from prompt.prompt_llm_eval import gen_eval_prompt

from argparse import ArgumentParser
from tqdm import tqdm

# set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model_id = (
    "/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.bfloat16},
    device_map="auto",
)


def extract_feedback_and_result(case_text: str):
    """Extract the single paragraph containing 'Feedback:' and return {'feedback': str, 'number': int|None}"""
    import re

    m = re.search(
        r"Feedback:\s*([\s\S]*?)(?=\n\s*Score:|\Z)", case_text, flags=re.IGNORECASE
    )
    if not m:
        return None
    fb_part = m.group(1).strip()

    num = None
    for pat in (
        r"\[RESULT\]\s*(\d+)",  # [RESULT] 1
        r"\[Score\s*(\d+)\]",  # [Score 3]
        r"\[\s*(\d+)\]",  # [2]
        r"(\d+)\.?\s*$",  # tail number 3 or 1.
    ):
        mm = re.search(pat, fb_part, flags=re.IGNORECASE)
        if mm:
            num = int(mm.group(1))
            break
    if num is None:
        mm = re.search(r"Score:\s*(\d+)", case_text, flags=re.IGNORECASE)
        if mm:
            num = int(mm.group(1))

    # clean the feedback text: remove all the marked form ([Score n], [RESULT] n, [n], tail number, and the trailing Score: n)
    cleaned = fb_part
    cleaned = re.sub(r"\[Score\s*\d+\]\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[RESULT\]\s*\d+\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*\d+\]\.?\s*", "", cleaned)  # [2]
    cleaned = re.sub(
        r"\n\s*Score:\s*\d+\s*$", "", cleaned, flags=re.IGNORECASE
    )  # trailing Score: 3 line
    # remove the isolated short number at the end (usually the number is the label, like " ... 3" or " ... 1."), but only remove the number with length <= 2 to avoid deleting the long number
    cleaned = re.sub(r"\s+\d{1,2}\.?\s*$", "", cleaned)
    # remove the extra whitespace and the trailing punctuation (keep the reasonable period unless it is the label)
    cleaned = cleaned.strip()
    cleaned = cleaned.rstrip(
        ".,;:"
    )  # if you want to keep the period, you can remove this line

    if cleaned == None or cleaned == "" or num == None:
        print("*" * 100)
        print(case_text)
        print("*" * 100)

    return {"feedback": cleaned, "number": num}


def load_data(dataset_path):
    # Dataset and output file configuration
    with open(dataset_path, "r", encoding="utf-8") as fp:
        json_data = json.load(fp)
    print(f"Loaded {len(json_data)} samples from {dataset_path}")
    return json_data


def inference(args, json_data):
    final_result = []
    for x in tqdm(range(len(json_data))):
        cur_data = json_data[x]
        messages = gen_eval_prompt(cur_data, args.rubric_name)
        outputs = pipeline(
            messages,
            max_new_tokens=1024,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            do_sample=False,
        )
        # print(cur_data['Ground_Truth_Solution'])
        # print("*"*100)
        # print(outputs[0]["generated_text"][-1])
        # print("*"*100)
        response = outputs[0]["generated_text"][-1]["content"]
        feedback_and_result = extract_feedback_and_result(response)
        cur_data[f"eval_{args.rubric_name}_result"] = {
            "prompt": messages,
            "result": response,
            "feedback": feedback_and_result["feedback"],
            "number": feedback_and_result["number"],
        }
        final_result.append(cur_data)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    return final_result


def inference_original(args, json_data):
    final_result = []
    for x in tqdm(range(len(json_data))):
        cur_data = json_data[x]
        for model_name, value in cur_data["anno_llm_responses"].items():
            response = value["response"]
            temp_data = {
                "conversation_history": cur_data["conversation_history"],
                "result": response,
            }
            messages = gen_eval_prompt(temp_data, args.rubric_name)
            outputs = pipeline(
                messages,
                max_new_tokens=1024,
                pad_token_id=pipeline.tokenizer.eos_token_id,
                do_sample=False,
            )

            response = outputs[0]["generated_text"][-1]["content"]
            feedback_and_result = extract_feedback_and_result(response)
            if "annotation_eval" not in value:
                value["annotation_eval"] = {}
            value["annotation_eval"][args.rubric_name] = {
                "result": response,
                "feedback": feedback_and_result["feedback"],
                "number": feedback_and_result["number"],
            }
        final_result.append(cur_data)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    return final_result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../data/MRBench/MRBench_V1_Mistral-7B-Instruct-v0.1.json",
    )
    parser.add_argument(
        "--mode", type=str, default="generate", choices=["generate", "original"]
    )

    args = parser.parse_args()
    input_file = args.dataset_file.split("/")[-1].replace(".json", "")
    args.output_file = os.path.join(
        os.path.dirname(args.dataset_file), input_file + "_llama_eval.json"
    )
    print("*" * 100)
    print(args)
    print("*" * 100)
    if args.mode == "generate":
        MRBench_V1_data = load_data("../data/MRBench/MRBench_V1.json")
        conversation_history_map = {
            data["conversation_id"] + data["Split"]: data["conversation_history"]
            for data in MRBench_V1_data
        }
        json_data = load_data(args.dataset_file)
        for data in json_data:
            data["conversation_history"] = conversation_history_map[
                data["conversation_id"] + data["Split"]
            ]
        rubric_name_list = [
            "mistake_identification",
            "mistake_location",
            "revealing_of_the_answer",
            "providing_guidance",
            "coherence",
            "actionability",
            "tutor_tone",
            "humanlikeness",
        ]

        for rubric_name in rubric_name_list:
            print(f"*" * 100)
            print(f"Evaluating {rubric_name}...")
            print(f"*" * 100)
            args.rubric_name = rubric_name
            json_data = inference(args, json_data)
        print(f"Evaluation completed.")
    else:
        MRBench_V1_data = load_data(args.dataset_file)
        rubric_name_list = [
            "mistake_identification",
            "mistake_location",
            "revealing_of_the_answer",
            "providing_guidance",
            "coherence",
            "actionability",
            "tutor_tone",
            "humanlikeness",
        ]

        for rubric_name in rubric_name_list:
            print(f"*" * 100)
            print(f"Evaluating {rubric_name}...")
            print(f"*" * 100)
            args.rubric_name = rubric_name
            MRBench_V1_data = inference_original(args, MRBench_V1_data)
