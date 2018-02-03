import numpy as np 
import pandas as pd 
import jieba.posseg
import sklearn
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import scipy.stats
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

jieba.load_userdict('dict.txt')
data = pd.read_csv('train.csv')
content_list = []
sentiment_words = []
# print(data['sentiment_word'])
for i in data['sentiment_word']:
    if type(i) == str:
        sentiment_words.append(i.strip().split(';')[0:-1])
    else:
        sentiment_words.append([])
theme_words = []
for i in data['theme']:
    if type(i) == str:
        theme_words.append(i.strip().split(';')[0:-1])
    else:
        theme_words.append([])
tags = []
for i in data['sentiment_anls']:
    if type(i) == str:
        tags.append(i.strip().split(';')[0:-1])
    else:
        tags.append([])
#df = pd.DataFrame({'content':content_list,'sentiment_word':sentiment_words})
#print(df.head())
#print(theme_words)
'''# make dictionary
pos = []
neg = []
non = []
theme = []
print(sentiment_words)
print(tags)
for sw_list,tag in zip(sentiment_words,tags):
    print(sw_list,tag)
    for w,t in zip(sw_list,tag):
        if t == '1':
            pos.append(w)
        if t == '-1':
            neg.append(w)
        if t == '0':
            non.append(w)
f = open('dict.txt','w')
#print(pos)
#print(neg)
#print(non)
for i in theme_words:
    for j in i:
        if j != 'NULL':
            theme.append(j)
print(theme)
for i in pos:
    f.write(i+'\n')
for i in neg:
    f.write(i+'\n')
for i in non:
    f.write(i+'\n')
for i in theme:
    f.write(i+'\n')
f.close()'''
def constructTextBylabeling(data,theme_words,sentiment_words,tags):
    train = []
    for con,thl,seml,tag in zip(data['content'],theme_words,sentiment_words,tags):
        i = jieba.posseg.cut(con)
        res = []
        tmp = []
        for p,q in i:
            tmp.append((p,q))
        for each in tmp:
            w = each[0]
            p = each[1]
            if w in thl:
                res.append((w,p,'t'))
            elif w in seml:
                t = tag[seml.index(w)]
                if t == '1':
                    res.append((w,p,'p'))
                elif t == '-1':
                    res.append((w,p,'n'))
                else:
                    res.append((w,p,'m'))
            else:
                res.append((w,p,'null'))
        train.append(res)
    return train  
data = constructTextBylabeling(data,theme_words,sentiment_words,tags)
#print(data)

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

def sent2labels(sent):
    return [label for token,tag,label in sent]
train = data[:int(len(data)*0.75)]
test = data[int(len(data)*0.75):]

X_train = [sent2features(s) for s in train]
y_train = [sent2labels(s) for s in train]

X_test = [sent2features(s) for s in test]
y_test = [sent2labels(s) for s in test]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
labels = list(crf.classes_)
#labels.remove('null')
print(labels)

y_pred = crf.predict(X_test)
f1 = metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)
print(y_pred)
print('ornginal:')
print(f1)

print(metrics.flat_classification_report(
    y_test, y_pred, labels=labels, digits=3
))

#joblib.dump(crf,'ornginal')


'''crf = sklearn_crfsuite.CRF()
params_space = {
    'algorithm':['lbfgs'],
    'max_iterations':[50,100,200,500,1000],
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
    'all_possible_transitions':[True,False],
    'all_possible_states':[True,False]
}

# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score, 
                        average='weighted', labels=labels)

# search
rs = RandomizedSearchCV(crf, params_space, 
                        cv=10, 
                        verbose=1, 
                        n_jobs=-1, 
                        n_iter=100, 
                        scoring=f1_scorer)
rs.fit(X_train, y_train)
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
joblib.dump(rs,'best_in_100')'''
'''rs = joblib.load('best_in_100')
y_pred = rs.predict(test)
f1 = metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)
print(f1)'''
print()












