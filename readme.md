# proposal
## 数据集处理
### 构建词表
已经完成于，见[此处](./vocab.py)，Vocab类将语料中出现的单词映射为一个唯一的编码。

数据可视化于[此](./DataDemo.ipynb)
### 构建dataset
### 寻找更多语料？
## 模型选择
### rnn
### gru
### lstm
### transformer
### 其他预训练模型？
## 模型训练
可阅读[这一篇](./单轮训练的基本操作.md)

对每个句子，其每个单词被编码为数字，每个数字被拓展为一个向量(通过nn.embedding实现)，每个输入的形状是(batch_size,num_steps,embedding_size),输出则是(batch_size,2),以交叉熵作为损失函数进行训练。