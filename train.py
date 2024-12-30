import torch
from torch import nn
import tools
import pandas
import data_process
import os
import argparse
from NN import *


'''Public Hyper Parameters'''
Batch_size = 64
use_mask = True
num_epochs = 50
model = 1
device = tools.try_gpu()
model_path = None#'models/model18.pth'
use_test_select = False
early_stop = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,required=True)
    parser.add_argument('--model_path',type=str,required=False,default=None)
    parser.add_argument('--model',type=int,required=False,default=None)
    args = parser.parse_args()
    name = args.name
    model_path = args.model_path if  model_path is None else model_path#有提前预设的就忽略命令行
    model = args.model if model is None else model
    torch.cuda.empty_cache()  # 释放未被引用的显存
    '''加载数据'''
    data  = pandas.read_csv('./motionClassify.csv')
    vocab = data_process.gen_vocab(data)
    data_train  =  data_process.gen_dataset(data[:40000],vocab,use_mask=use_mask)
    data_test = data_process.gen_dataset(data[40000:],vocab,use_mask=use_mask)
    #ramdom_data_test = data_process.gen_dataset(tools.select(data[40000:],500),vocab)

    train_iter = torch.utils.data.DataLoader(data_train,Batch_size,shuffle=True,num_workers=8)
    test_iter = torch.utils.data.DataLoader(data_test,Batch_size,shuffle=True,num_workers=8)
    '''模型选择'''
    if model ==1:
        '''BERT的参数'''
        num_hiddens, num_layers, dropout =  768, 12, 0.1#num_hidden 其实是embedding_size或者说embedding_dim
        lr = 2e-5
        key_size=query_size=value_size=ffn_num_input=norm_shape=num_hiddens
        ffn_num_hiddens, num_heads =  64,12


            
        net = transformerEncoder(len(vocab),key_size,query_size,value_size,num_hiddens,
                                norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers
                                ,dropout)

        optimizer = torch.optim.Adam(net.parameters(),lr,weight_decay=0.01)
    elif  model==2:
        lr = 0.1
        net = LSTM(num_embeddings=len(vocab),embedding_dim=256,
                   hidden_size=256,device=device,use_mask=use_mask)
        if model_path:
            net.load_state_dict(torch.load(model_path))
        optimizer = torch.optim.SGD(net.parameters(),lr)
    
    
    '''训练和测试'''
    logs = tools.train(net,train_iter,device,optimizer,torch.nn.CrossEntropyLoss(),
                       epoch=num_epochs,test_iter=test_iter if use_test_select else None
                       ,use_mask=use_mask,early_stop=early_stop)#这次先取消test_iter,看看基于batch的优化好不好

    accuracy=tools.test(net,test_iter,device,use_mask=use_mask)

    torch.save(net.state_dict(),'models/'+name+'.pth')
    os.system(f'cp train.py models/{name}.py')#这一行在windows系统会失效，因为cp不是windows的命令
    with open('models/'+name+'.txt','w') as f:
        f.write(f'accuracy = {accuracy}\n')
        for log in logs:
            f.write(log+'\n')
    torch.cuda.empty_cache()