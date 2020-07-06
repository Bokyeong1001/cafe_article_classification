#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import os
import time
import datetime
from tensorflow.python.platform import flags
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
komoran = Komoran("EXP")

def tokenize(doc):
    return komoran.get_morphes_by_tags(doc, tag_list=['NNP', 'NNG'])

#kkma=Kkma()
selected_words=[]

data_path = 'test_data.csv'

data = pd.read_table(data_path, sep=",", encoding="ANSI")
data=np.array(data)
for row in data:
    a=str(row[0])+" "+str(row[1])
    row[0]=a

now = datetime.datetime.now()
print(now)
test_docs = [(tokenize(row[0])) for row in data]
# JSON 파일로 저장
with open('validate_docs.json', 'w', encoding="ANSI") as make_file:
    json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")
now = datetime.datetime.now()
print(now)

test_docs = tokenize(data)

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
    print(words[0])
    print(len(words))
    print("selected_words read")
except:
    print("NO selected_words")


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

test_data = [term_frequency(d) for d in test_docs]
test_data=np.array(test_data)

new_model = keras.models.load_model('result_model.h5')
new_model.summary()
predictions = new_model.predict(test_data)
count = 0
k=0

#file_name="zresult_test1_긍정" #->0
#file_name="zresult_test1_부정" #->1
#file_name="zresult_test1_중립" #->2
file_name="zresult_test1_무관" #->3
#file_name="zresult_test1_타부" #->4


f = open("result/"+file_name+".txt", 'w')

for i in predictions:
    if np.argmax(i) == 0:   ##테스트 할때 마다 바꿔줘야함##
        count += 1
    res=""
    if np.argmax(i)==1:
        res=str(data[k])
        res=res+" : 유관\n\n"
    f.write(res)
    k=k+1
#print(np.argmax(predictions[0]))
f.close()

print(count/len(predictions)*100,"%")
