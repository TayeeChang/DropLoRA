import argparse
import sys
import os
os.environ['VLLM_INSTALL_PUNICA_KERNELS'] = "1"
import torch
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="", default="output-conversations-8b/3e-4/v0-20250403-131130/checkpoint-2234-merged")
parser.add_argument("--data_path", type=str, default="datasets")
parser.add_argument('--sub_task', nargs='+', help='', default=['conversations'])
parser.add_argument('--dataset_split', type=str, default="test", help='')
parser.add_argument('--output_file', type=str, default="output/llama3_8b/conv/conv_r32_a64_e2_droplora_inner.jsonl", help="")
parser.add_argument("--batch_size", type=int, default=40000, help="")
parser.add_argument('--temperature', type=float, default=0.0, help="")
parser.add_argument('--top_p', type=float, default=1, help="")
parser.add_argument('--max_tokens', type=int, default=2048, help="")
args = parser.parse_args()


stop_tokens = []
sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, stop=stop_tokens)
llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count()) # 
print()
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


with torch.no_grad():
    completions = llm.generate(batch_dataset_query, sampling_params)
    
for query, completion, answer, task in zip(batch_dataset_query, completions, batch_dataset_answer, batch_dataset_task):
    with open(args.output_file, 'a') as f:
        json.dump({'type': task, 'query': query, 'output': completion.outputs[0].text, 'answer': answer}, f)
        f.write('\n')