'''
此文件中出现的模型都服务于情感二分类，其输入形状都是(batch_size,max_num_steps)
定义网络时选择use_mask可以输入valid_len实现掩蔽，形状是(batch_size,)，里面的元素是每句话的有效长度
'''
import torch
from torch import nn
import MyTransformer
class LSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, device, num_layers=1, use_mask=False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask =use_mask
        self.embedding = nn.Embedding(num_embeddings, embedding_dim,padding_idx=0 if use_mask else None)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, dropout=0.5, batch_first=True)#增加了droput层和双向
        self.ff = nn.Linear(2 * hidden_size, 2)  # 双向LSTM的输出需要乘以2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

    def forward(self, X,valid_len=None):
        h0 = torch.zeros(self.num_layers * 2, X.shape[0], self.hidden_size).to(self.device)  # 隐藏状态初始化为0
        c0 = torch.zeros(self.num_layers * 2, X.shape[0], self.hidden_size).to(self.device)
   
        X = self.embedding(X)
        output, _ = self.lstm(X if not self.use_mask else torch.nn.utils.rnn.pack_padded_sequence(X,valid_len.to('cpu'),batch_first=True, enforce_sorted=False)
                              , (h0, c0))
        if self.use_mask:
            output,_ = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
            

        # 使用最后一个时间步的输出进行分类
        output_pooled = torch.max(output, dim=1)[0]  # 这里使用最大池化
        return self.ff(output_pooled)


class transformerEncoder(nn.Module):
    def __init__(self,vocab_size,key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout,use_mask=False):
        super().__init__()
        self.encoder = MyTransformer.TransformerEncoder(
            vocab_size, key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
            num_layers, dropout,padding_idx=0 if use_mask else None)
        self.dropout = nn.Dropout(0.1)##改成0.1
        self.dense = nn.Linear(num_hiddens,2)
    def forward(self,X,valid_len=None):
        enc_X = self.encoder(X,valid_len)
        output, _ = torch.max(enc_X,dim=1)
        return self.dense(self.dropout(output))# 加入最大池化