import collections

def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens) # .items()访问所有键值对
        # counter : word-frequency dict
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True) # 词频从大到小的排序
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens #此attribute把索引映射为字符串，用列表实现
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)} # 此dict把字符串映射为索引，注意enumerate从零开始计数
        # i = 0
        # print(type(self._token_freqs))
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:# 此句是为了防止_token_freqs里的词和reserved_tokens重复
                self.idx_to_token.append(token)
                ''' if i < 10:
                    i = i+1
                    print(token)'''
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # 尾部分词的索引

    # __function允许以运算符的方式调用class 的 method
    def __len__(self):
        return len(self.idx_to_token)   

    def __getitem__(self, tokens):#这个功能把输入语料映射为数字索引
        # 在tokens为单独的词时停止递归
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)# 如果找不到就返回self.unk， .get()不会引发keyerror异常
        return [self.__getitem__(token) for token in tokens] # 递归，返回的索引列表形状和传入列表形状相同

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property  # 只传self的方法可以@property变成一个attribute
    def unk(self):  # 未知词元的索引为0
        return 0


def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    return collections.Counter(tokens) # 返回一个字典，key：元素； value ： 出现次数

if __name__ == '__main__':
    pass