# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:16:12 2017

@author: HTX
"""

import pandas as pd
import os
import numpy as np

'''
###########################################
读取股评
###########################################
'''
path='C:/Users/tianxiang-hu/Desktop/HTX/HTX/数据集/目标数据集/shuffle后标记csv/新建文件夹'
stockRemarkPd=pd.DataFrame()
files=os.listdir(path)

for file in files:
    
    if not os.path.isdir(file):
        
        stockRemark1=pd.read_csv(path+'/'+file,encoding="GBK")
        
        stockRemark1=stockRemark1[(stockRemark1['mark']==0)|(stockRemark1['mark']==1)]
        stockRemarkPd=stockRemarkPd.append(stockRemark1)
        

'''
#############################################
对股评进行onehot编码
#############################################
'''
dic=pd.read_hdf('dic.h5','df')
charSet=set(dic.index)
def sent2Num(x,maxLen):
    x=[i for i in x if i in charSet]
    x=x[:maxLen]+[' ']*max(0, maxLen-len(x))
    return list(dic[x])
maxLen=25
stockRemarkPd['sent2Num'] = stockRemarkPd['content'].apply(lambda x: sent2Num(x, maxLen))
stockRemarkPd=stockRemarkPd.drop(['content'],axis=1)
from sklearn.utils import shuffle
print(len(stockRemarkPd[(stockRemarkPd['mark']==0)]))
print(len(stockRemarkPd[(stockRemarkPd['mark']==1)]))
stockRemarkPd=shuffle(stockRemarkPd)
pos=stockRemarkPd[(stockRemarkPd['mark']==1)]
neg=stockRemarkPd[(stockRemarkPd['mark']==0)][0:len(pos)]
print(len(neg))
print(len(pos))
stockRemarkPd=pos.append(neg)

stockRemarkPd=shuffle(stockRemarkPd)


x=np.array(list(stockRemarkPd['sent2Num']))
    #print(x)
xtrain=x[0:int(len(x)*0.8)]

xtest=x[int(len(x)*0.8):]

    
   
y=np.array(list(stockRemarkPd['mark']))
ytrain=y[0:int(len(x)*0.8)]
ytest=y[int(len(x)*0.8):]

from keras.layers import Input, Embedding, Dense,Flatten
from keras.models import Model
from keras.layers import Conv1D,MaxPooling1D



inputRaw=Input(shape=(maxLen,), dtype='int32')
embeddingShared= Embedding(output_dim=64, input_dim=len(dic), input_length=maxLen,name="Embedding")(inputRaw)
covShared1=Conv1D(filters=64, kernel_size=6, padding='same', activation='relu',name="Cov1")(embeddingShared)
maxShared1=MaxPooling1D(pool_size=2,name="Maxpooling1")(covShared1)
covShared2=Conv1D(filters=128, kernel_size=6, padding='same', activation='relu',name="Cov2")(maxShared1)
maxShared2=MaxPooling1D(pool_size=2,name="Maxpooling2")(covShared2)
covShared3=Conv1D(filters=64, kernel_size=6, padding='same', activation='relu',name="Cov3")(maxShared2)
maxShared3=MaxPooling1D(pool_size=2,name="Maxpooling3")(covShared3)
covShared4=Conv1D(filters=128, kernel_size=6, padding='same', activation='relu',name="Cov4")(maxShared3)
maxShared4=MaxPooling1D(pool_size=2,name="Maxpooling4")(covShared4)
out=Flatten()(maxShared4)
    
    
nlpModel=Model(inputRaw,out,name="modelShared")

inputStock = Input(shape=(25,), dtype='int32')


outStock=nlpModel(inputStock)
denseStock1=Dense(16,activation='relu')(outStock)
denseStock2=Dense(1,activation='sigmoid')(denseStock1)    
model = Model(inputs=inputStock,outputs=denseStock2)  
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.load_weights('mymodel.h5', by_name=True)
model.fit(xtrain,ytrain,epochs=8,batch_size=128)
scores=model.evaluate(xtest,ytest)
print(scores)

#model=load_model('mymodel.h5')
yprob=model.predict(xtest)
yclass=np.around(yprob)
#print(yclass)


from sklearn.metrics import precision_recall_fscore_support

print(precision_recall_fscore_support(yclass,ytest))


