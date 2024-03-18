import os
import json
import torch
import argparse
import jsonlines
from tqdm import tqdm
import torch
import torch.distributed as dist
from itertools import chain
from math import ceil

from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_DICT_NONE = {
    "prompt_input": (
        "{instruction}\n{input}\n"
    ),
    "prompt_no_input": (
        "{instruction}\n"
    ),
}

def build_inputs(tokenizer, query: str, assistant, meta_instruction="", ):
        if tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = tokenizer.bos_token
        if meta_instruction:
            prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        whole_prompt = prompt + f"{assistant}<|im_end|\n>"
        return tokenizer([prompt], return_tensors="pt"), tokenizer([whole_prompt], return_tensors="pt")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/alpaca_data/alpaca_data.json')
    parser.add_argument("--save_path", type=str, default='debug.jsonl')
    parser.add_argument("--model_name_or_path", type=str, default='gpt2')
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default='none', help='none')
    args = parser.parse_args()
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(model, whole_input_ids):

    try:
        with torch.no_grad(): 
            outputs = model(whole_input_ids, labels=whole_input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()

    # except:
    except Exception as e:
        raise e
        return 0, 0

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(model, input_ids):

    try:
        
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()
    
    except Exception as e:
        raise e
        return 0, 0


def main():

    args = parse_args()
    
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir='../cache', output_hidden_states=True)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache')

    # model.eval()
    # init_dist(launcher='slurm', backend='nccl')
    
    # torch.cuda.set_device(global_rank)
    print(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # global_rank = int(os.environ.get("RANK"), 0)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # print(local_rank)
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir='../cache', output_hidden_states=True, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache', trust_remote_code=True)

    data = []
    with open(args.data_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    
    # data: list of dict
    # with open(args.data_path, "r") as f:
    #     data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file

    with open(args.save_path, "r") as file:
        exsisting_num =  sum(1 for _ in file)
    sampled_data = sampled_data[exsisting_num:]

    all_data = sampled_data
    ori_length = len(all_data)
    pad_num = ceil(len(all_data) / world_size) * world_size - len(all_data)
    for i in range(pad_num):
        all_data.append(all_data[i])
    print(f'========================={len(all_data)}============================')

    datas = all_data[local_rank::world_size]
    print(f">>> running {len(datas)} docs")


    if args.prompt == 'none':
        prompt_no_input = PROMPT_DICT_NONE["prompt_no_input"]
        prompt_input = PROMPT_DICT_NONE["prompt_input"]

    ifd_scores = []
    for i in tqdm(range(len(datas))):

        data_i = datas[i]
        # print(data[i])
        # instruct_i = data_i['instruction']
        # output_i = data_i['output']
        
        '''
        for multi-dialogue
        '''
        # ifds = []
        # for i in range(0, len(data_i), 3):
        #     meta_instruction = data_i[i]['content']
        #     query = data_i[i+1]['content']
        #     assistant = data_i[i+2]['content']
        
        #     inputs, whole_inputs = build_inputs(tokenizer, query, assistant, meta_instruction)
        #     inputs = {k: v.cuda() for k, v in inputs.items() if torch.is_tensor(v)}
        #     whole_inputs = {k: v.cuda() for k, v in whole_inputs.items() if torch.is_tensor(v)}
        #     # also add end-of-assistant token in eos token id to avoid unnecessary generation
        #     # eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

        #     ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(model, whole_inputs['input_ids'])
        #     ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(model, inputs['input_ids'])

        #     temp_data_i = {}
        #     temp_data_i['ppl'] = [0,ppl_out_alone,0,ppl_out_condition]
        #     temp_data_i['loss'] = [0,loss_out_alone,0,loss_out_condition]
        #     ifd = ppl_out_condition/ppl_out_alone
        #     ifds.append(ifd)

        # ifd_scores.append(sum(ifds)/len(ifds))
        
        '''
        for simple-dialogue without meta
        '''
        meta_instruction = None
        query = data_i[0]['content']
        assistant = data_i[1]['content']
        
        '''
        for simple_dialogue with meta
        '''
        # meta_instruction = data_i[0]['content']
        # query = data_i[1]['content']
        # assistant = data_i[2]['content']
        
        inputs, whole_inputs = build_inputs(tokenizer, query, assistant, meta_instruction)
        inputs = {k: v.cuda() for k, v in inputs.items() if torch.is_tensor(v)}
        whole_inputs = {k: v.cuda() for k, v in whole_inputs.items() if torch.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        # eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

        ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(model, whole_inputs['input_ids'])
        ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(model, inputs['input_ids'])

        temp_data_i = {}
        temp_data_i['ppl'] = [0,ppl_out_alone,0,ppl_out_condition]
        temp_data_i['loss'] = [0,loss_out_alone,0,loss_out_condition]
        ifd = ppl_out_condition/ppl_out_alone
        ifd_scores.append(ifd)
        # data_i[0]['ifd_score'] = ifd
        # data_i[0]['ifd_score'] = ifd
        # with open(args.save_path, "a") as file:
        #     file.write(json.dumps(temp_data_i) + '\n')
        # with jsonlines.open(args.save_path, 'a', flush=True) as file:
        #     file.write(data_i)
    tmp = [None] * world_size
    # dist.all_gather_object(tmp, complexity_outputs)
    dist.all_gather_object(tmp, ifd_scores)
    outputs = list(chain(*zip(*tmp)))[:ori_length]
    if local_rank == 0:
        with jsonlines.open(args.save_path, 'a', flush=True) as file:
            for data, ifd_score in zip(all_data, outputs):
                data[0]['ifd_score'] = ifd_score
                file.write(data)
    
    print('Done: Data Analysis:',args.data_path)

if __name__ == "__main__":
    main()