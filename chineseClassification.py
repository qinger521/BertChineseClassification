from cProfile import label
from cgitb import text
import torch
import datasets
from datasets import load_dataset
from transformers import BertTokenizer
import transformers

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

dataset = Dataset("D:\Tencent\HuggingFace\DataSet\ChnSentiCorp")
print(len(dataset))
print(dataset[0])

# 加载tokenizer
token = BertTokenizer.from_pretrained("D:\Tencent\HuggingFace\\bert-base-chinese")

# 定义批处理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,truncation=True,padding='max_length',max_length=500,return_tensors='pt',return_length=True)

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels

# 定义数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=True)

# 加载预训练模型
pretrain_model = transformers.BertModel.from_pretrained("D:\Tencent\HuggingFace\\bert-base-chinese")

for param in pretrain_model.parameters():
    param.requires_grads = False

# 定义下游任务
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrain_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out

model = Model()
optimizer = transformers.AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
model.train()
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        print("step : {}\tloss : {}\taccuracy : {}".format(i, loss.item(), accuracy))
    if i == 300:
        break


