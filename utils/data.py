import json
from torch.utils.data import Dataset
import tiktoken
import torch
file_path = '/home/gpt/dataset/chinese_modern_poetry/chinese_poems.jsonl'
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
