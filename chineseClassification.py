from cProfile import label
from cgitb import text
import torch
import datasets
from datasets import load_dataset
from transformers import BertTokenizer

# 创建数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, name) -> None:
        self.dataset = datasets.load_from_disk(name)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label

dataset = Dataset("D:\Tencent\HuggingFace\DataSet\ChnSentiCrop")
print(len(dataset))
print(dataset[0])

# 加载tokenizer
token = BertTokenizer.from_pretrained('bert-base-chinese')