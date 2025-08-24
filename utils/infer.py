from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse
import sys
import os
from datasets import load_dataset, concatenate_datasets


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="", default="/common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/")
parser.add_argument("--data_path", type=str, default="datasets")
parser.add_argument('--sub_task', nargs='+', help='', default=['commonsense'])
parser.add_argument('--dataset_split', type=str, default="test", help='')
parser.add_argument('--output_file', type=str, default="output/llama3_8b/cms/cms_r32_a64_e3_droplora_dynamic_unmerge.jsonl", help="")
parser.add_argument("--batch_size", type=int, default=40000, help="")
parser.add_argument('--temperature', type=float, default=0.0, help="")
parser.add_argument('--top_p', type=float, default=1, help="")
parser.add_argument('--max_tokens', type=int, default=2048, help="")
args = parser.parse_args()

base_model = AutoModelForCausalLM.from_pretrained("/common-data/pretrained_models/llama3-8b/LLM-Research/Meta-Llama-3-8B/")
adapter = PeftModel.from_pretrained(base_model, "output-cms3-8b-dynamic/3e-4/v0-20250331-184232/checkpoint-3459")


def batch_data(data_list, batch_size=1):
        n = len(data_list) // batch_size
        batch_data = []
        for i in range(n-1):
            start = i * batch_size
            end = (i+1)*batch_size
            batch_data.append(data_list[start:end])

        last_start = (n-1) * batch_size
        last_end = sys.maxsize
        batch_data.append(data_list[last_start:last_end])
        return batch_data


if args.sub_task is None:
    dataset = load_dataset(args.data_path, split=args.dataset_split)
else:
    all_test_dataset = []
    for task in args.sub_task:
        ds = load_dataset(args.data_path, data_dir=task, split=args.dataset_split)
        print(f"{args.data_path}/{task}/{args.dataset_split}")
        for k,v in ds[0].items():
            print("-"*100)
            print(k,end=':\t')
            print(v)
        print("+"*100)
        all_test_dataset.append(ds)
        
    dataset = concatenate_datasets(all_test_dataset)
    

batch_dataset_query = dataset["instruction"]
batch_dataset_answer = dataset["output"]
batch_dataset_task = dataset["type"]

# print()
