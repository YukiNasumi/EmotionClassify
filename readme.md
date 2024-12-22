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
### [softmax+ff](./TrainWithFF.ipynb)
完全没有效果，但可以阅读这一篇了解训练的流程
### rnn
### gru
### [lstm](./lstm.ipynb)
纯粹的lstm也几乎没有效果
增加了以下策略：
- 双向
- dropout
- 最大池化
- 隐藏状态用0初始化

准确率达到82%

还不知道是哪一个策略起了效果，需要对比文件中的两个自定义lstm控制变量研究一下

12.15已经新增任务于[lstm文件](./lstm.ipynb),见文档的最后一个cell
### transformer
2024.12.22 新增准备文件
### 其他预训练模型？
GPT2(small)尝试，取最后一个logits再连接一个一个二分头
需要hugging face 库 `pip3 install transformer`
## 模型训练
可阅读[这一篇](./单轮训练的基本操作.md)

对每个句子，其每个单词被编码为数字，每个数字被拓展为一个向量(通过nn.embedding实现)，每个输入的形状是(batch_size,num_steps,embedding_size),输出则是(batch_size,2),以交叉熵作为损失函数进行训练。