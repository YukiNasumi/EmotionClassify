# 数据分析

## 总体分析

之前的实验已经证明加入最大池化对LSTM模型有着飞跃性的提升，现在想知道如何训练才能达到更大的准确率。

加入掩蔽机制，将填充的词忽略以后模型性能反而下降，原因未知

注意到batch_size设置为64时模型损失持续下降，未出现收敛。batch_size设置的越大，批次的数量就越小，模型更新的次数就越少，相应的模型的泛化(generalisation)能力会更强，防止出现过拟合(overfit)。模型不收敛说明还可继续更新参数，在训练数据集不扩充的情况下可以减小batch_size。

将batch_size设置为4，8，16，32，进行单轮训练，发现batch_size越小准确率越高(batch_size为4和8 时表现其实差不多)最好的batch_size=4，在4轮训练后准确率达到[89%](models/LSTM/model16.txt)。

## 下一步实验（如果还有时间的话）

- 继续增加训练轮数，看看各项指标的变化
- 加入掩蔽机制再对照实验
