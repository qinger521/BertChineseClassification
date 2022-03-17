import torch
from datasets import load_dataset
from transformers import BertTokenizer
import transformers
import datasets

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset = datasets.load_from_disk("D:\\Tencent\\HuggingFace\\DataSet\\ChnSentiCorp")

        def f(data):
            return len(data["text"]) > 30
        self.dataset = dataset.filter(f)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]["text"]

dataset = Dataset("train")

# 加载tokenizer
token = BertTokenizer.from_pretrained("D:\\Tencent\\HuggingFace\\bert-base-chinese")

def collate_fn(data):
    data = token.batch_encode_plus(batch_text_or_text_pairs=data, truncation=True, padding='max_length', max_length=30, return_tensors='pt', return_length=True)
    input_ids = data["input_ids"]
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    # 将每句话的第十五个词mask掉
    labels = input_ids[:, 15].reshape(-1).clone()
    input_ids[:, 15] = token.get_vocab()[token.mask_token]

    return input_ids, attention_mask, token_type_ids, labels

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, drop_last=True)

pretrained = transformers.BertModel.from_pretrained("D:\\Tencent\\HuggingFace\\bert-base-chinese")

for param in pretrained.parameters():
    param.requires_grad = False

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(768, token.vocab_size)
        torch.nn.init.xavier_normal(self.decoder.weight)
        torch.nn.init.constant(self.decoder.bias, 0.)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids, attention_mask, token_type_ids)
        out = self.decoder(out.last_hidden_state[:, 15])
        #out = out.softmax(dim=-1)
        return out

model = Model()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(5):
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        out = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 50 == 0:
            out = out.argmax(dim=1)
            print("out shape {}".format(out.shape))
            print("label shape {}".format(labels.shape))
            accuracy = (out == labels).sum().item() / len(labels)
            print("epoch : {}\tstep : {}\tloss : {}\taccuracy : {}".format(epoch, i, loss, accuracy))
