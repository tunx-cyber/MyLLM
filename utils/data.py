import json
from torch.utils.data import Dataset
import tiktoken
import torch
file_path = '/home/gpt/dataset/chinese_modern_poetry/chinese_poems.jsonl'

######一次性加载######
def read_json_file(file_path = '/home/gpt/dataset/chinese_modern_poetry/chinese_poems.jsonl'):
    """
    读取JSON文件并返回数据
    :param file_path: JSON文件路径
    :return: 返回字典或列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]  # 返回字典或列表
    return data

#poems
def process_data(data):
    string_data = []
    for item in data:
        if isinstance(item, dict):
            string_data.append(item['content'])
        elif isinstance(item, list):
            string_data.extend(item)
    print(f"数据长度: {len(string_data)}")
    return string_data

#滑动窗口
class PoemDataset(Dataset):
    def __init__(self, str_data, max_length=1024,stride=1,tokenizer = tiktoken.get_encoding("gpt2")):
        self.input_ids = []
        self.target_ids = []
        for item in str_data:
            token_ids = tokenizer.encode(item)
            for i in range(0,len(token_ids)-max_length, stride):
                input_chunk = token_ids[i : i+max_length]
                target_chunk = token_ids[i+1 : i+max_length+1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
def get_data():
    data = read_json_file(file_path)
    data = process_data(data)
    return data

#######流式加载######
from torch.utils.data import IterableDataset
def sliding_window(text,tokenizer, window_size=2048, stride=1024):
    tokens = tokenizer.encode(text)
    for start in range(0, len(tokens), stride):#对齐window size 所以会到最后一个token
        end = start + window_size
        window = tokens[start:end]
        
        # 处理填充
        pad_len = max(0, window_size - len(window))
        input_ids = window + [tokenizer.eot_token] * pad_len  # GPT-2的pad_token是<|endoftext|>
        attention_mask = [1] * len(window) + [0] * pad_len
        
        # 生成目标（左移一位）
        labels = input_ids[1:] + [tokenizer.eot_token]  # 最后一个位置预测pad
        
        yield {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "labels": torch.tensor(labels)
        }

# 还可以改进点：1. mmap映射文件，2. 通过多进程读取数据
class LLMIterableDataset(IterableDataset):
    def __init__(self, file_path=file_path, tokenizer=tiktoken.get_encoding("gpt2"), window_size=1024, stride=1024//4):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                line = json_obj['content']
                yield from sliding_window(line, self.tokenizer, self.window_size, self.stride)