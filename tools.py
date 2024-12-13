import torch
from tqdm import tqdm
def train(net,train_iter,device,optimizer,criterion):
    net.train()
    net.to(device)
    for i,(X,y) in tqdm(enumerate(train_iter),total=len(train_iter)):
        X = X.to(device)
        
        y = y.to(device)
        
        optimizer.zero_grad()
        y_hat = net(X)
        loss = criterion(y_hat,y)
        loss.backward()
        if (i+1)%100 == 0:
            tqdm.write(f'batch{i+1},loss = {loss.item()}')
        optimizer.step()
        
def test(net,test_iter,device):
    net.eval()
    net.to(device)
    with torch.no_grad():
        tp = 0
        cnt = 0
        for i,(X,y) in tqdm(enumerate(test_iter),total=len(test_iter)):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            tp += (y==y_hat.argmax(dim=1)).sum()
            cnt += y.shape[0]
            
    print(f'accuracy = {tp/cnt}')

    