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
    def __init__(self,vocab,data,max_len=600):
        self.vocab = vocab
        self.review = [ s.split()[:max_len] for s in data['review']]#truncnate
        #print(torch.tensor(vocab[self.review])[0])
        self.tokens =  pad_sequence([torch.tensor(token) for token in vocab[self.review]],batch_first=True,padding_value=0)# 将每个句子编码后填充
        self.label = data['label']
        self.max_len = max_len
    
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self,index):
        return self.tokens[index],torch.tensor(self.label[index])

def gen_dataset(data,max_len=600):
    return emotionDataset(gen_vocab(data),data,max_len)
    
    