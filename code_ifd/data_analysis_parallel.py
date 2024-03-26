import torch
import json
import argparse
import jsonlines
import os
from math import ceil
from tqdm import tqdm
import torch
import torch.distributed as dist
from math import ceil
import typing as tp
from pathlib import Path
from mmengine.dist import get_rank, get_world_size, init_dist
from mmengine import MMLogger
from transformers import AutoTokenizer, AutoModelForCausalLM
from filelock import FileLock, SoftFileLock
from collections import defaultdict
import pandas as pd


logger = None
im_start = '[UNUSED_TOKEN_146]'
im_end = '[UNUSED_TOKEN_145]'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("data_root", type=str)
    parser.add_argument("save_root", type=str)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--launcher", type=str, default='none')
    parser.add_argument("--log_file", type=str, default='./ifds.log')
    parser.add_argument("--lock_file", type=str, default='ifd_lock.lock')
    args = parser.parse_args()
    return args


def build_system_instruction(meta_instruction: str):
    return f"""{im_start}system\n{meta_instruction}{im_end}\n"""


def build_user_instruction(instruction: str):
    return f"""{im_start}user\n{instruction}{im_end}\n{im_start}assistant\n"""


def build_assistant_instruction(answer: str):
    return f"""{answer}{im_end}\n"""


class DataInfo(tp.TypedDict):
    source_path: Path
    saved_path: Path
    content: list
    ifd_score: float
    idx: int


class IFDInputs(tp.NamedTuple):
    inputs: tp.Tuple[tp.List[int], tp.List[int]]  # data, label
    whole_inputs: tp.Tuple[tp.List[int], tp.List[int]]  # data, label


def is_multi_turn(data: list):
    return len(data) > 3


def tokenize_inputs(instruction, answer, tokenizer, max_seq_len=8192):
    instruction_token = tokenizer.encode(instruction, return_tensors="pt").cuda()
    answer_token = tokenizer.encode(answer, return_tensors="pt").cuda()

    answer_token = answer_token[:, :max_seq_len]
    whole_input_ids = torch.cat([instruction_token, answer_token], dim=1)
    whole_input_labels = torch.cat([torch.full_like(instruction_token, -100), answer_token], dim=1)

    strip_length = whole_input_ids.shape[1] - max_seq_len
    if strip_length > 0:
        whole_input_ids = whole_input_ids[strip_length:]
        whole_input_labels = whole_input_labels[strip_length:]
    
    whole_inputs = (whole_input_ids, whole_input_labels)
    inputs = (answer_token, answer_token)
    return IFDInputs(inputs=inputs, whole_inputs=whole_inputs)



def build_inputs(data_info: DataInfo, tokenizer, max_seq_len=8192, max_rounds=8) -> tp.List[IFDInputs]:
    rounds = data_info['content']
    model_inputs = []
    for idx, single_round in enumerate(rounds):
        if single_round['role'] != 'assistant':
            continue
        content_str_list = []
        for pre_round in rounds[:idx]:
            if pre_round['role'] == 'assistant':
                content_str_list.append(build_assistant_instruction(pre_round['content']))
            elif pre_round['role'] == 'user':
                content_str_list.append(build_user_instruction(pre_round['content']))
            elif pre_round['role'] == 'system':
                content_str_list.append(build_system_instruction(pre_round['content']))
            else:
                return None
        content_str_list.append(build_assistant_instruction(single_round['content']))
        instruction = ''.join(content_str_list[:-1])
        answer = ''.join(content_str_list[-1])
        ifd_input_list = tokenize_inputs(instruction, answer, tokenizer, max_seq_len)
        if ifd_input_list:
            model_inputs.append(ifd_input_list)
        if len(model_inputs) == max_rounds:
            break
    return model_inputs


def get_raw_data_info(dataset_root: Path, dataset_saved_root: Path, num: int = -1) -> tp.List[DataInfo]:
    world_size = get_world_size()
    rank = get_rank()

    data_infos = []
    origin_path = dataset_root / 'processed'
    for source_path in origin_path.rglob("**/*.jsonl"):
        saved_path = dataset_saved_root / source_path.relative_to(dataset_root)
        with jsonlines.open(source_path) as f:
            for idx, data in enumerate(f):
                if num != -1 and idx >= num:
                    break
                data_infos.append(DataInfo(source_path=source_path, saved_path=saved_path, content=data, ifd_score=None, idx=idx))
    data_infos = data_infos[rank::world_size]
    return data_infos


def inference(ifd_inputs: tp.List[IFDInputs], model):
    ifd_scores = []
    for ifd_input in ifd_inputs:
        inputs, inputs_label = ifd_input.inputs
        whole_inputs, whole_input_label = ifd_input.whole_inputs
        with torch.inference_mode():
            inputs_loss = model(inputs, labels=inputs_label).loss
            whole_inputs_loss = model(whole_inputs, labels=whole_input_label).loss
        inputs_perplexity = torch.exp(inputs_loss).cpu().item()
        whole_inputs_perplexity = torch.exp(whole_inputs_loss).cpu().item()
        ifd_scores.append(whole_inputs_perplexity / inputs_perplexity)

    return sum(ifd_scores) / len(ifd_scores)


def save_result(cache_file):
    file_handlers = {}
    f = open(cache_file)
    for data in f:
        data = json.loads(data)
        saved_path = data['saved_path']
        Path(saved_path).parent.mkdir(exist_ok=True, parents=True)
        if saved_path not in file_handlers:
            file_handlers[saved_path] = jsonlines.open(saved_path, 'w')
        content = data['content']
        content[0]['IFD_score'] = data['ifd_score']
        file_handlers[saved_path].write(content)
    f.close()
    for f in file_handlers.values():
        f.close()


def save_result_to_excel(cache_file):
    def contnt_to_str(content):
        content_str = ''
        for single_round in content:
            content_str += f"{single_round['role']}\n"
            content_str += f"{single_round['content']}\n\n"
        return content_str

    df_data_dict = defaultdict(dict)
    f = open(cache_file)
    for data in f:
        data = json.loads(data)
        saved_path = data['saved_path']
        if 'content' not in df_data_dict[saved_path]:
            df_data_dict[saved_path]['content'] = []
            df_data_dict[saved_path]['ifd_score'] = []

        content = data['content']
        content_str = contnt_to_str(content)
        ifd_score = data['ifd_score']


        df_data_dict[saved_path]['content'].append(content_str)
        df_data_dict[saved_path]['ifd_score'].append(ifd_score)

    for saved_path, df_data in df_data_dict.items():
        saved_path = Path(saved_path)
        saved_path.parent.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(df_data)
        df.to_excel(saved_path.with_suffix('.xlsx'), index=False)

    


def parse_cache_file(cache_file: Path):
    data_infos = []
    # jsonline could failed to parse the cache file, do not why
    with open(cache_file) as f:
        for line in f.readlines():
            data_infos.append(json.loads(line))
    cache_mapping = defaultdict(set)
    for data_info in data_infos:
        cache_mapping[data_info['source_path']].add(data_info['idx'])
    return cache_mapping


def is_processed(data_info: DataInfo, cache_mapping):
    source_path_str = str(data_info['source_path'])
    return source_path_str in cache_mapping and data_info['idx'] in cache_mapping[source_path_str]


def main():
    global logger
    args = parse_args()
    local_rank = 0
    init_dist(launcher=args.launcher)
    local_rank = int(os.getenv('LOCAL_RANK'))
    torch.cuda.set_device(local_rank)

    logger = MMLogger('IFDs', log_file=args.log_file)

    lock = SoftFileLock('~/.cache/' + args.lock_file)
    data_root = Path(args.data_root)
    save_root = Path(args.save_root)
    save_root.mkdir(exist_ok=True, parents=True)

    cache_file = save_root / '.cachefile'

    f = jsonlines.open(cache_file, mode='a', flush=True)

    if cache_file.exists():
        cache_mapping = parse_cache_file(cache_file)
    else:
        cache_mapping = defaultdict(set)

    dataset_paths = list(data_root.iterdir())
    # hardcode to place the Belle at the end
    for path in dataset_paths.copy():
        if 'Belle' == path.name:
            dataset_paths.remove(path)
            dataset_paths.append(path)

    print("=======!!!!!!!LOADING MODEL!!!!!!!=======")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).cuda()
    for dataset_path in dataset_paths:
        dataset_saved_root = save_root / dataset_path.name
        data_infos = get_raw_data_info(dataset_path, dataset_saved_root, args.num)
        if local_rank == 0:
            data_infos = tqdm(data_infos, desc=f'Scoring {dataset_path.name} IFDs...')
        for data_info in data_infos:
            if is_processed(data_info, cache_mapping):
                continue
            ifd_inputs = build_inputs(data_info, tokenizer, args.max_seq_len, args.max_rounds)
            if not ifd_inputs:
                logger.warning(f"Failed to build inputs for {data_info['source_path']} {data_info['idx']}")
                data_info['ifd_score'] = None
            else:
                ifd_score = inference(ifd_inputs, model)
                data_info['ifd_score'] = ifd_score

            data_info = {k: str(v) if isinstance(v, Path) else v for k, v in data_info.items()}
            with lock:
                f.write(data_info)
    f.close()


if __name__ == "__main__":
    main()
