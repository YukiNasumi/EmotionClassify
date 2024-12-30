from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch.utils.data import Dataset,DataLoader
import pandas
from tqdm import tqdm
import tools
import argparse

class IMDBdataest(Dataset):
    def __init__(self,texts,labels):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inputs = self.tokenizer(texts,truncation=True, max_length=1024,padding=True)
        self.ids = self.inputs['input_ids']
        self.attention_masks = self.inputs['attention_mask']
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return (torch.tensor(self.ids[index]),torch.tensor(self.attention_masks[index])),torch.tensor(self.labels[index])

def train(net,train_iter,device,num_epochs,lr):
    net.train()
    net.to(device)
    logs=[]
    optimzier = torch.optim.Adam(net.parameters(),lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1,num_epochs+1):
        batch = 1
        for (X,atten_mask),label in tqdm(train_iter):
            optimzier.zero_grad()
            X=X.to(device)
            atten_mask = atten_mask.to(device)
            label = label.to(device)
            y_hat = net(input_ids=X, attention_mask=atten_mask)
            loss = criterion(y_hat.logits,label)
            loss.backward()
            if batch%100==0:
                log = f'epoch{epoch},batch{batch} loss={loss.item()}'
                logs.append(log)
            optimzier.step()
            batch+=1
    return logs

def test(net,test_iter,device):
    net.eval()
    net.to(device)
    with torch.no_grad():
        tp = 0
        cnt = 0
        for (X,atten),y in tqdm(test_iter):
            X = X.to(device)
            atten = atten.to(device)
            y = y.to(device)
            y_hat = net(input_ids=X, attention_mask=atten).logits
            tp += (y==y_hat.argmax(dim=1)).sum()
            cnt += y.shape[0]
    print(f'accuracy = {tp/cnt}')
    return tp/cnt

device = tools.try_gpu()
num_epochs = 10
batch_size = 64
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,required=True)
    args = parser.parse_args()
    name = args.name
    data  = pandas.read_csv('./motionClassify.csv')
    train_texts, train_labels = list([' '.join(s.split()[:1024]) for s in  data[:40000]['review']]),list(data[:40000]['label'])
    test_texts, test_labels = list([' '.join(s.split()[:1024]) for s in data[40000:]['review']]),list(data[40000:]['label'])

    
    train_dataset = IMDBdataest(train_texts, train_labels)
    test_dataset = IMDBdataest(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    model.config.pad_token_id = train_dataset.tokenizer.pad_token_id

    logs = train(model,train_loader,device,num_epochs=num_epochs,lr=2e-5)
    torch.save(model.state_dict(),'models/'+name+'.pth')
    accuracy = test(model,test_loader,device)
    
    
    with open('models/'+name+'.txt','w') as f:
        f.write(f'accuracy = {accuracy}\n')
        for log in logs:
            f.write(log+'\n')
