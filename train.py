import torch
from torch import nn
import tools
import pandas
import data_process
import MyTransformer
import argparse
class transformerEncoder(nn.Module):
    def __init__(self,vocab_size,key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout):
        super().__init__()
        self.encoder = MyTransformer.TransformerEncoder(
            vocab_size, key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
            num_layers, dropout)
        self.dropout = nn.Dropout(0.1)##改成0.1
        self.dense = nn.Linear(num_hiddens,2)
    def forward(self,X):
        enc_X = self.encoder(X)
        output, _ = torch.max(enc_X,dim=1)
        return self.dense(self.dropout(output))# 加入最大池化
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,required=True)
    args = parser.parse_args()
    name = args.name
    torch.cuda.empty_cache()  # 释放未被引用的显存

    data  = pandas.read_csv('./motionClassify.csv')
    vocab = data_process.gen_vocab(data)
    data_train  =  data_process.gen_dataset(data[:40000],vocab)
    data_test = data_process.gen_dataset(data[40000:],vocab)
    #ramdom_data_test = data_process.gen_dataset(tools.select(data[40000:],500),vocab)
    Batch_size = 64
    train_iter = torch.utils.data.DataLoader(data_train,Batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(data_test,Batch_size,shuffle=True)

    num_hiddens, num_layers, dropout =  768, 12, 0.1#num_hidden 其实是embedding_size或者说embedding_dim
    lr, num_epochs, device = 2e-5, 50, tools.try_gpu()
    key_size=query_size=value_size=ffn_num_input=norm_shape=num_hiddens
    ffn_num_hiddens, num_heads =  64,12


        
    net = transformerEncoder(len(vocab),key_size,query_size,value_size,num_hiddens,
                            norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers
                            ,dropout)

    optimizer = torch.optim.Adam(net.parameters(),lr,weight_decay=0.01)

    logs = tools.train(net,train_iter,device,optimizer,torch.nn.CrossEntropyLoss(),epoch=num_epochs,test_iter=test_iter)

    accuracy=tools.test(net,test_iter,device)

    torch.save(net.state_dict(),'models/'+name+'.pth')
    with open('models/'+name+'.txt','w') as f:
        f.write(f'accuracy = {accuracy}\n')
        for log in logs:
            f.write(log+'\n')
    torch.cuda.empty_cache()