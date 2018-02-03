import numpy as np 
import pandas as pd 
import jieba.posseg
import sklearn
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import scipy.stats
from sklearn.externals import joblib

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

jieba.load_userdict('dict.txt')
data = pd.read_csv('test.csv')
def word2features(sent,i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2]    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2]
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:postag': postag1,
            '-1:postag[:2]': postag1[:2]
        })
    else:
        features['EOS'] = True
                
    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def handleTestData(data):
    test = []
    for con in data['content']:
        i = jieba.posseg.cut(con)
        res = []
        tmp = []
        for p,q in i:
            	res.append((p,q))
        test.append(res)
    return test 
test = handleTestData(data)
test = [sent2features(s) for s in test]
model = joblib.load('ornginal')
pred = model.predict(test)
print(pred)
theme = []
sentiment_word = []
tags = []

for s,p in zip(data['content'],pred):
	ttmp = []
	stmp = []
	tgtmp = []
	sentmp = []
	sent = jieba.posseg.cut(s)
	for w,l in sent:
		sentmp.append(w)
	for w,t in zip(sentmp,p):
		if t == 't':
			ttmp.append(w)
		if t == 'p':
			stmp.append(w)
			tgtmp.append(1)
		if t == 'm':
			stmp.append(w)
			tgtmp.append(0)
		if t == 'n':
			stmp.append(w)
			tgtmp.append(-1)
	theme.append(ttmp)
	sentiment_word.append(stmp)
	tags.append(tgtmp) 
with open('result.csv','w') as f:
	f.write('row_id,content,theme,sentiment_word,sentiment_anls'+'\n')
	for i,sen in zip(range(1,10001),data['content']):
		f.write(str(i)+','+sen+',')
		for j in theme[i-1]:
			f.write(j+';')
		f.write(',')
		for k in sentiment_word[i-1]:
			f.write(k+';')
		f.write(',')
		for l in tags[i-1]:
			f.write(str(l)+';')
		f.write('\n')
