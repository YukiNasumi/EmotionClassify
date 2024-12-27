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
                             for idx, token in enumerate(self.idx_to_token)}  # 此dict把字符串映射为索引，注意enumerate从零开始计数
        
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:  # 防止特殊词元与其他词元重复
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0  # '<unk>' 的索引为0

    @property
    def pad(self):
        return self.token_to_idx.get('<pad>',0)  # 返回 '<pad>' 的索引

    @property
    def bos(self):
        return self.token_to_idx.get('<bos>',0)  # 返回 '<bos>' 的索引

    @property
    def eos(self):
        return self.token_to_idx.get('<eos>',0)  # 返回 '<eos>' 的索引



def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    return collections.Counter(tokens) # 返回一个字典，key：元素； value ： 出现次数

if __name__ == '__main__':
    pass