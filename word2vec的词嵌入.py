#!/usr/bin/env python
# coding: utf-8



import codecs
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
import gensim

data=pd.read_excel('data/复旦大学中文文本分类语料.xlsx','sheet1') 

import jieba
jieba.enable_parallel() #并行分词开启
data['文本分词'] = data['正文'].apply(lambda i:jieba.cut(i) )
data['文本分词'] =[' '.join(i) for i in data['文本分词']]



lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.分类.values)


xtrain, xvalid, ytrain, yvalid = train_test_split(data.文本分词.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)


X=data['文本分词']
X=[i.split() for i in X]


model = gensim.models.Word2Vec(X,min_count =5,window =8,size=100)   # X是经分词后的文本构成的list，也就是tokens的列表的列表
embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))

print('Found %s word vectors.' % len(embeddings_index))



#该函数会将语句转化为一个标准化的向量（Normalized Vector）

  
def sent2vec(s):
    import jieba
    jieba.enable_parallel() #并行分词开启
    words = str(s).lower()
    #words = word_tokenize(words)
    words = jieba.lcut(words)
    words = [w for w in words if not w in stwlist]
    #words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            #M.append(embeddings_index[w])
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())




stwlist=[line.strip() for line in open('data/停用词汇总.txt',
'r',encoding='utf-8').readlines()]




# 对训练集和验证集使用上述函数，进行文本向量化处理
xtrain_w2v = [sent2vec(x) for x in tqdm(xtrain)]
xvalid_w2v = [sent2vec(x) for x in tqdm(xvalid)]

def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


# 基于word2vec特征在一个简单的Xgboost模型上进行拟合
clf = xgb.XGBClassifier(nthread=10, silent=False)
clf.fit(xtrain_w2v, ytrain)
predictions = clf.predict_proba(xvalid_w2v)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))




# 基于word2vec特征在一个简单的Xgboost模型上进行拟合
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
clf.fit(xtrain_w2v, ytrain)
predictions = clf.predict_proba(xvalid_w2v)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

