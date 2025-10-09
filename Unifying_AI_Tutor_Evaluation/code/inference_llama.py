import transformers
import torch
import numpy as np
import random
import os
import json
from prompt.prompt import prompt_MathDial, prompt_Bridge
from argparse import ArgumentParser
from tqdm import tqdm

# set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model_id = "/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
prompt_MathDial = prompt_MathDial()
prompt_Bridge = prompt_Bridge()

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.bfloat16},
    device_map="auto",
)


def load_data(args):
    # Dataset and output file configuration
    dataset_path = os.path.join(args.dataset_file)
    with open(dataset_path, "r", encoding="utf-8") as fp:
        json_data = json.load(fp)
    print(f"Loaded {len(json_data)} samples from {args.dataset_file}")
    return json_data


def inference(args, json_data):
    final_result = []
    for x in tqdm(range(len(json_data))):
        cur_data = json_data[x]
        if cur_data['Data'] == "MathDial":
            messages = prompt_MathDial.gen_prompt(cur_data)
        elif cur_data['Data'] == "Bridge":
            messages = prompt_Bridge.gen_prompt(cur_data)
        else:
            raise ValueError(f"Invalid data source: {cur_data['Data']}")
        

        outputs = pipeline(
            messages,
            max_new_tokens=1024,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            # greedy_search
            do_sample=False
        )
        # print(cur_data['Ground_Truth_Solution'])
        # print("*"*100)
        # print(outputs[0]["generated_text"][-1])
        # print("*"*100)
        response = outputs[0]["generated_text"][-1]['content']
        result_data = {
            'conversation_id': cur_data['conversation_id'],
            'Split': cur_data['Split'],
            'prompt': messages,
            'result': response
        }
        final_result.append(result_data)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="../data/MRBench/MRBench_V1.json")
    args = parser.parse_args()
    args.output_file = os.path.join(os.path.dirname(args.dataset_file), "MRBench_V1_Meta-Llama-3.1-8B-Instruct.json")
    print("*"*100)
    print(args)
    print("*"*100)
    json_data = load_data(args)
    inference(args, json_data)