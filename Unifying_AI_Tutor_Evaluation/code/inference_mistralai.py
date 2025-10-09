import torch
import numpy as np
import random
import os
import json
from prompt.prompt import prompt_MathDial, prompt_Bridge
from argparse import ArgumentParser
from tqdm import tqdm

from modelscope import AutoModelForCausalLM, AutoTokenizer


# set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model_id = "/u/xfeng4/.cache/modelscope/hub/models/mistralai/Mistral-7B-Instruct-v0.1"
prompt_MathDial = prompt_MathDial()
prompt_Bridge = prompt_Bridge()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set pad_token to eos_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.to(device)
model.eval()

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
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True, add_generation_prompt=True)
        input_ids = encodeds.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
            # full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
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
    args.output_file = os.path.join(os.path.dirname(args.dataset_file), "MRBench_V1_Mistral-7B-Instruct-v0.1.json")
    print("*"*100)
    print(args)
    print("*"*100)
    json_data = load_data(args)
    # json_data = json_data[:10]
    inference(args, json_data)