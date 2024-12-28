import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
import pandas
from tqdm import tqdm
import argparse
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, torch.tensor(self.labels[idx])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,required=True)
    args = parser.parse_args()
    name = args.name

    batch_size = 64
    data  = pandas.read_csv('./motionClassify.csv')
    train_texts, train_labels = list(data[:40000]['review']),list(data[:40000]['label'])
    test_texts, test_labels = list(data[40000:]['review']),list(data[40000:]['label'])



    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 二分类任务

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    train_dataset = IMDbDataset(train_encodings, train_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # 5. 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # 6. 微调BERT模型
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device('cpu')
    model.to(device)


    # 7. 训练过程
    logs = []
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        logs.append(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # 8. 评估模型
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(),'models/'+name+'.pth')
    with open('models/'+name+'.txt','w') as f:
        f.write(f'accuracy = {accuracy}\n')
        for log in logs:
            f.write(log+'\n')
