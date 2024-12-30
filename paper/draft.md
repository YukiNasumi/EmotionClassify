# 基于多种循环神经网络和注意力模型的情感分类模型

## Abstract

**情感分析** （也称为**观点挖掘**或 **情感 AI** ）是使用自然语言处理、文本分析、计算语言学和生物识别技术来系统地识别、提取、量化和研究情感状态和主观信息。情感分析广泛应用于客户反馈材料（如评论和调查回复）、在线和社交媒体以及医疗保健材料，应用范围从营销到客户服务再到临床医学。

情感分类研究最早可以追溯到20世纪90年代，随着互联网的普及，人们开始尝试从文本数据中提取情感信息。从简单的规则词典方法到传统机器学习方法（支持向量机等），到近些年的深度学习方法进而到预训练大语言模型的引入，情感分类问题的研究方法和方案在不断地迭代更新。本文的关注点侧重于深度学习方法，在pytorch的框架下从零开始实现多种神经网络方法，探究了不同网络，不同参数的设计对模型性能、实验结果的影响。

本文采用IMDB数据集作为语料库和训练、测试集，自行构建词表将语料中的单词映射为索引，包装成为数据集
本文从最简单的全连接神经网络开始，逐步复现了基本的循环神经网络（Recurrent Neural Networks）、门控循环单元（Gated Recurrent Unit）、长短期记忆网络（Long Short-Term Memory）等经典神经网络。

此外，结合时下炙手可热的生成式预训练大语言模型（Generative Pretrained Language Model），本文实现了位置编码,点积注意力、自注意力机制、多头注意力，加入残差连接构成经典的seq2seq（序列到序列）架构transformer的编码器(\ref{https://arxiv.org/abs/1706.03762})。通过编码器的堆叠构建了类似BERT的结构，在云服务器上完成了训练。

最后，受到产业界微调训练方式(\ref{https://arxiv.org/pdf/2303.06135})的影响，本文尝试了多种预训练模型的微调。效果？

**key words** RNN,LSTM,PrLMs,Transformer
## Introduction(放到历史后面）
写前置工作：
1. [词表](../vocab.py)
2. 词嵌入 nn.Embedding 介绍一下embedding，可以写很多，[参考](https://zhuanlan.zhihu.com/p/114538417)
## related work
情感分类问题的研究
1. 早期阶段（20世纪90年代中后期）
  
  - **代表性工作**：  
    1997年，Pang等人首次提出了对电影评论进行情感分类的任务，标志着情感分类研究的起步。  
  - **方法**：  
    主要使用**规则和词典**方法，如基于情感词典的匹配方式，通过对情感词汇表（如正面、负面词汇）的统计分析实现简单分类。  

---

2. 传统机器学习阶段（2000年 - 2010年）
  这一阶段情感分类开始使用传统的机器学习方法，摆脱了单纯依赖词典匹配的方式。  
  - **关键技术**：  
    - **Naive Bayes（朴素贝叶斯）**  
    - **Support Vector Machine（支持向量机，SVM）**  
    - **Logistic Regression（逻辑回归）**  
  - **特征提取方式**：  
    - **TF-IDF（词频-逆文档频率）**  
    - **N-gram**（多元组词汇特征）  
    - **词袋模型（Bag of Words, BoW）**  
  - **代表性工作**：  
    2002年，Pang等人使用SVM对影评文本进行情感分析，证明了传统机器学习方法在情感分类中的有效性。  

---

3. 深度学习阶段（2013年 - 至今） ：  
  随着深度学习的发展，情感分类模型逐渐引入了神经网络，尤其是RNN、LSTM和CNN模型。  
  - **关键技术**：  
    - **Recurrent Neural Networks（RNN）**  
    - **Long Short-Term Memory（LSTM）**  
    - **Convolutional Neural Networks（CNN）**  
    - **Gated Recurrent Unit（GRU）**  
  - **突破**：  
    2014年，Kim提出的**Text-CNN**模型在多种文本分类任务中取得了较好的表现，为情感分析提供了一种新的思路。  

---

4. 预训练语言模型阶段（2018年 - 至今）
- **BERT和GPT的出现**：  
  2018年后，预训练语言模型（如BERT和GPT）彻底改变了自然语言处理（NLP）的格局。  
  - **代表性模型**：  
    - **BERT（Bidirectional Encoder Representations from Transformers）**  
    - **GPT（Generative Pre-trained Transformer）**  
    - **RoBERTa**、**XLNet**等变体  
  - **特点**：  
    - 这些模型通过大规模语料库预训练，再在下游情感分类任务中进行微调，大幅提升了分类精度。  
  - **应用示例**：  
    使用BERT进行细粒度情感分析，能够识别文本中的细微情感差异，例如中性、积极或消极情绪的不同程度。  

---

## 数据集和语料库
IMDB数据集
## Experiment
### baseline模型：简单的全连接层
### 经典循环神经网络
#### baseline rnn 50%
#### lstm 
accuracy-batch_size
loss-batch_size
A/B contrast : use_mask =True/False



rnn 介绍参考[这一篇](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html)
### lstm
### TransformerEncoder
#### small
#### middle(800M参数)
### 预训练模型
