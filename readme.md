# proposal
## 数据集处理
### 构建词表(先自己实现了，后期可以用现成的试试看)
已经完成于，见[此处](./vocab.py)，Vocab类将语料中出现的单词映射为一个唯一的编码。

数据可视化于[此](./DataDemo.ipynb)
### 构建dataset
对于Dataset对象，用索引方式得到一个tuple(input,label)

已经实现于[此](./data_process.py)
使用方法见[DatasetDemo](./DatasetDemo.ipynb)
#### padding
每个句子的长度不一样，编码后要填充为一样的长度，统计发现大部分句子长度在200到600之间，所以在600处截断。
#### masking（可选，先实现训练，后期可以再加）
防止填充的部分影响模型训练，故采用掩蔽(masking)
1. rnn等循环神经网络的掩蔽策略
2. transformer的掩蔽
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