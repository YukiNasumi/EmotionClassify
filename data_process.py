import vocab
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence


def gen_vocab(data):
    lines = [l for l in data['review']]
    lines = list(map(word_tokenize,lines))
    corpus = [word  for l in lines for word in l]# 语料库
    return vocab.Vocab(corpus)


class emotionDataset(Dataset):
    def __init__(self,vocab,data,max_len=600,use_mask=False):
        self.use_mask = use_mask
        self.vocab = vocab
        self.review = [ s.split()[:max_len] for s in data['review']]#truncnate
        self.val_len = [len(s.split()) if len(s.split())<=max_len else max_len for s in data['review']]
        #print(torch.tensor(vocab[self.review])[0])
        self.tokens =  pad_sequence([torch.tensor(token) for token in vocab[self.review]],batch_first=True,padding_value=0)# 将每个句子编码后填充
        self.label = [label for label in data['label']]
        self.max_len = max_len
    
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self,index):
        if not self.use_mask:
            return self.tokens[index],torch.tensor(self.label[index])
        else:
            return (self.tokens[index],torch.tensor(self.val_len[index])),torch.tensor(self.label[index])

def gen_dataset(data,vocab = None,max_len=600,use_mask=False):
    if not vocab:
        vocab = gen_vocab(data)
    return emotionDataset(vocab,data,max_len,use_mask)
    
    