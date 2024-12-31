import torch
from tqdm import tqdm
import seq2seq
import random
def train(net,train_iter,device,optimizer,criterion,
          epoch=1,test_iter=None,use_mask=False,early_stop=False):
#    xavier_init(net)
    net.train()
    net.to(device)
    min_accu = 0.5
    logs=[]
    last_loss = 10
    for j in range(epoch):
        net.train()
        if not use_mask:
            for i,(X,y) in tqdm(enumerate(train_iter),total=len(train_iter)):
                X = X.to(device)
                
                y = y.to(device)
                
                optimizer.zero_grad()
                y_hat = net(X)
                loss = criterion(y_hat,y)
                optim = last_loss - loss.item()
                
                loss.backward()
                seq2seq.grad_clipping(net,1)
                if (i+1)%100 == 0:
                    log = f'epoch{j+1},batch{i+1},loss = {loss.item()}'
                    logs.append(log)
                    if early_stop:
                        if optim < -0.3:##反向优化太大则早停,注意在optimizer优化之前停止
                            logs.append('early_stop,ignore the last loss')
                            net.load_state_dict(torch.load('models/cache.pth'))
                            return logs
                        else:
                            torch.save(net.state_dict(),'./models/cache.pth')
                optimizer.step()
        else:##这是被证明没有用的mask
            for i,(Xs,y) in tqdm(enumerate(train_iter),total=len(train_iter)):
                X,valid_len = [x.to(device) for x in Xs]
                
                y = y.to(device)
                
                optimizer.zero_grad()
                y_hat = net(X,valid_len)
                loss = criterion(y_hat,y)
                loss.backward()
                seq2seq.grad_clipping(net,1)
                if (i+1)%100 == 0:
                    log = f'epoch{j+1},batch{i+1},loss = {loss.item()}'
                    logs.append(log)
                optimizer.step()
        if test_iter:
            net.eval()
            accuracy = test(net,test_iter,device,use_mask=use_mask)
            if accuracy > min_accu:
                torch.save(net.state_dict(),'./models/cache.pth')
    if test_iter:
        net.load_state_dict(torch.load('models/cache.pth'))
    return logs
        
def test(net,test_iter,device,use_mask=False):
    net.eval()
    net.to(device)
    with torch.no_grad():
        tp = 0  # True Positives
        tn = 0  # True Negatives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        cnt = 0
        if not use_mask:
            for i,(X,y) in tqdm(enumerate(test_iter),total=len(test_iter)):
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                preds = y_hat.argmax(dim=1)  # 预测类别，通常是 0 或 1
            
                # 计算真正例 (True Positive)
                tp += ((preds == 1) & (y == 1)).sum().item()
                # 计算假正例 (False Positive)
                fp += ((preds == 1) & (y == 0)).sum().item()
                # 计算真负例 (True Negative)
                tn += ((preds == 0) & (y == 0)).sum().item()
                # 计算假负例 (False Negative)
                fn += ((preds == 0) & (y == 1)).sum().item()

                
        else:
            for i,(Xs,y) in tqdm(enumerate(test_iter),total=len(test_iter)):
                (X,valid_len) = [x.to(device) for x in Xs]
                y = y.to(device)
                y_hat = net(X,valid_len)
                y_pred = y_hat.argmax(dim=1)
    
                # 计算 True Positives, False Positives, True Negatives, False Negatives
                tp += ((y_pred == 1) & (y == 1)).sum().item()  # 预测为正且实际为正
                fp += ((y_pred == 1) & (y == 0)).sum().item()  # 预测为正但实际为负
                tn += ((y_pred == 0) & (y == 0)).sum().item()  # 预测为负且实际为负
                fn += ((y_pred == 0) & (y == 1)).sum().item()  # 预测为负但实际为正
    print(f'accuracy = {tp/cnt}')
    print(f'precision = {tp/(tp+fp)}')
    print(f'recall = {tp/(tp+fn)}')
    return tp/cnt

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')   

def xavier_init(net):
    assert isinstance(net,torch.nn.Module)
    for module in net.modules():
        if(isinstance(module,torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)


def select(data,k):##生成随机数据集
    idxs = random.sample(list(range(len(data))),k)
    return {'review':[data['review'][i] for i in idxs],
            'sentiment':[data['sentiment'][i] for i in idxs],
            'label':[data['label'][i] for i in idxs]}