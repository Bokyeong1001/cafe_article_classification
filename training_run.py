
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import os
import time
import datetime
import tensorflow as tf
import numpy as np
import sys
from tensorflow import keras
import nltk
import pandas as pd
import re
import random
import json
import nltk
import datetime
import h5py
from PyKomoran import *
from keras.callbacks import EarlyStopping

komoran = Komoran("EXP")

def term_frequency(doc):
    res=[]
    for keyword in selected_words:
        res.append(doc.count(keyword))
    return res

def make_output(points):
    results = np.zeros((len(points),1))

    for idx, point in enumerate(points):
        if(point!=4):
            results[idx]=1
        else:
            results[idx]=0
    return results

def divide(x, y, train_prop):
    random.seed(1234)
    x = np.array(x)
    y = np.array(y)
    tmp = np.random.permutation(np.arange(len(x)))
    x_tr = x[tmp][:round(train_prop * len(x))]
    y_tr = y[tmp][:round(train_prop * len(x))]
    x_te = x[tmp][-(len(x)-round(train_prop * len(x))):]
    y_te = y[tmp][-(len(x)-round(train_prop * len(x))):]
    return x_tr, x_te, y_tr, y_te

def tokenize(doc):
    try:
        return komoran.get_morphes_by_tags(doc, tag_list=['NNP','NNG'])
    except:
        return "0"
        

selected_words=[]

data_path = 'learning_data.csv'
corpus = pd.read_table(data_path, sep=",", encoding="ANSI")
train_data = np.array(corpus)

for row in train_data:
    a=str(row[0])+" "+str(row[1])
    row[0]=a

if os.path.isfile('learning_docs.json'):
    with open('learning_docs.json', encoding="ANSI") as f:
        train_docs = json.load(f)
else:
    now = datetime.datetime.now()
    print(f"tokenize start: {now}")
    train_docs = [(tokenize(row[0]), row[2]) for row in train_data]
    # JSON 파일로 저장
    with open('learning_docs.json', 'w', encoding="ANSI") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    now = datetime.datetime.now()
    print(f"tokenize end: {now}")

train_docs = np.array(train_docs)
print(f"train data shape:{train_docs.shape}")
        
try: 
    f = open("selected_words.txt", 'r')
    words=f.readline()
    words = words.replace("[","")
    words = words.replace("]","")
    words = words.replace("'","")
    words = words.replace(" ","")
    words = words.split(",")
    selected_words=[]
    i=0
    for word in words:
        selected_words.append(words[i])
        i=i+1
    f.close()
    print("selected_words read")
except:
    now = datetime.datetime.now()
    print(f"select words start: {now}")
    tokens=[]
    for d in train_docs:
        d=np.array(d)
        k=0
        if(d[0]!="0"):
            for j in d[0]:
                if(len(d[0][k])>0):
                    tokens.append(d[0][k])
                k=k+1
    text = nltk.Text(tokens, name='NMSC')
    f = open("selected_words.txt", 'w')
    selected_words = [f[0] for f in text.vocab().most_common(10000)]
    f.write(str(selected_words))
    f.close()
    now = datetime.datetime.now()
    print(f"select words end: {now}")

print(f"selected_words length:{len(selected_words)}")

train_x=[]

now = datetime.datetime.now()
print(f"make train data start: {now}")

for d, _ in train_docs:
    train_x.append(term_frequency(d))
    
train_y = [c for _, c in train_docs]
y = make_output(train_y)



now = datetime.datetime.now()
print(f"make train data end: {now}")

# divide dataset into train/test set
print("divide start")
x_train, x_test, y_train, y_test = divide(train_x,y,train_prop=0.9)
print("divide end")

x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

model = keras.Sequential([
    keras.layers.Dense(128 ,input_shape=(x_train.shape[1],),activation='sigmoid'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128 ,activation='sigmoid'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128 ,activation='sigmoid'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128 ,activation='sigmoid'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128 ,activation='sigmoid'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128 ,activation='sigmoid'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping()  
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

model.save('my_model/result_model')
model.summary()
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\n테스트 정확도:', test_acc)
