
import pandas as pd
import re
data = pd.read_csv('IMDB Dataset.csv')
data['label'] = data['sentiment'].apply(lambda x : 1 if x=='positive' else 0)
def remove_html_tags(reviews):#去除链接
    return re.sub(r'<[^<]+?>', '', reviews)
data['review'] = data['review'].apply(lambda x : x.lower())
def remove_url(text):
    return re.sub(r'http[s]?://\S+|www\.\S+', '', text)
data['review'] = data['review'].apply(remove_html_tags)
data['review'] = data['review'].apply(remove_url)
import string
data['review'] = data['review'].apply(lambda x : x.translate(str.maketrans('', '', string.punctuation)))
# maketrans(a,b,c),把a映射为b并删除c的映射表(字典)，.translate()是字符串自带的方法 
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               # emoticons
                               u"\U0001F300-\U0001F5FF"
                               # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"
                               # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"
                               # flags (105)
                               u"\U00002702-\U00002780"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
data['review'] = data['review'].apply(remove_emoji)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''# 下载nltk的停用词列表和分词器数据
nltk.download('stopwords')
nltk.download('punkt')'''

def remove_stopwords(text):
    # 获取英语停用词列表
    stop_words = set(stopwords.words('english'))
    # 将文本分词
    word_tokens = word_tokenize(text)
    # 过滤掉停用词
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    # 将过滤后的词重新组合成字符串
    return ' '.join(filtered_text)
data['review'] = data['review'].apply(remove_stopwords)
if __name__ == '__main__':
    print(data.head())
    data.to_csv('../motionClassify/motionClassify.csv',index=False)