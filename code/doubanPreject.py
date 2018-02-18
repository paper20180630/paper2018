# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:32:59 2017

@author: HTX
"""
import pandas as pd
import numpy as np
import os


'''
提取豆瓣评论数据并处理 getBookRemarkData(bookPath) 
bookpath为豆瓣评价文件存放位置
'''
def getBookRemarkData(bookPath):
    content=[]
    rating=[]
    length=[]
    files=os.listdir(bookPath)
    for file in files:
        if not os.path.isdir(file):
            bookRemarkData=open(bookPath+"/"+file,encoding= 'utf-8')
            for i in bookRemarkData:
                content.append(i.split("\", \"")[0][13:])
                rating.append(i.split("\", \"")[1][9:-3])
                length.append(len(i.split("\", \"")[0][13:]))
    
    bookRemarkAll=pd.DataFrame({"content":content,"rating":rating,"length":length})
    bookRemarkAll=bookRemarkAll[bookRemarkAll["length"]<25]
    pos=bookRemarkAll[bookRemarkAll["rating"]=='力荐']
    print(len(pos))
    
    neg=bookRemarkAll[(bookRemarkAll["rating"]=='很差')|(bookRemarkAll["rating"]=='较差')]
    pos=pos.sample(n=len(neg))
    pos['mark']=1
    print(len(pos))
    neg['mark']=0
    
    print(len(neg))
    bookRemarkPd=pos.append(neg).drop(['rating'],axis=1)
    
    return bookRemarkPd


'''
One Hot编码处理 getOneHot(remarkPd,maxLen,minCount,dic)
'''

def makeDic(bookRemarkPd,maxLen,minCount):
    remarkAll=bookRemarkPd
    content=''.join(remarkAll['content'])
    dic=pd.Series(list(content)).value_counts()
    dic=dic[dic >= minCount]
    dic[:]=range(len(dic))
    
    return dic
    
def sent2Num(x,maxLen,dic):
    charSet=set(dic.index)
    x=[i for i in x if i in charSet]
    x=x[:maxLen]+[' ']*max(0, maxLen-len(x))
    return list(dic[x])
    
def getOneHot(remarkPd,maxLen,minCount,dic):
    remarkPd['sent2Num']=remarkPd['content'].apply(lambda x: sent2Num(x,maxLen,dic))
    remarkPd=remarkPd.drop(['content','length'],axis=1)
    return remarkPd    


'''
深度学习建模
'''


def main():
    
    bookPath="C:/Users/tianxiang-hu/Desktop/HTX/HTX/douban"
    bookRemarkPd = getBookRemarkData(bookPath)
    
    minCount=30
    maxLen=25
    dic=makeDic(bookRemarkPd,maxLen,minCount)
    
    bookRemarkPd=getOneHot(bookRemarkPd,maxLen,minCount,dic)
    from sklearn.utils import shuffle
    bookRemarkPd=shuffle(bookRemarkPd)

  
    x=np.array(list(bookRemarkPd['sent2Num']))
    xtrain=x[0:int(len(x)*0.8)]
    xtest=x[int(len(x)*0.8):]
   
    y=np.array(list(bookRemarkPd['mark']))
    ytrain=y[0:int(len(x)*0.8)]
    ytest=y[int(len(x)*0.8):]
     
    from keras.models import Sequential
    from keras.layers import Dense,Embedding,Flatten
    from keras.layers import Conv1D,MaxPooling1D
    
    model=Sequential()
    model.add(Embedding(len(dic),64,input_length=maxLen,name="Embedding"))
    model.add(Conv1D(filters=64,kernel_size=6,padding='same',activation='relu',name="Conv1"))
    model.add(MaxPooling1D(pool_size=2,name="Maxpooling1"))

    model.add(Conv1D(filters=128,kernel_size=6,padding='same',activation='relu',name="Conv2"))
    model.add(MaxPooling1D(pool_size=2,name="Maxpooling2"))

    model.add(Conv1D(filters=64,kernel_size=6,padding='same',activation='relu',name="Conv3"))
    model.add(MaxPooling1D(pool_size=2,name="Maxpooling3"))

    model.add(Conv1D(filters=128,kernel_size=6,padding='same',activation='relu',name="Conv4"))
    model.add(MaxPooling1D(pool_size=2,name="Maxpooling4"))
      
    model.add(Flatten())
    model.add(Dense(32,activation='relu',name="Dense1"))   
    model.add(Dense(1,activation='sigmoid',name="Dense2"))    
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    
    
    model.fit(xtrain,ytrain,epochs=5,batch_size=128)
    print("**************")
    scores=model.evaluate(xtest,ytest,verbose=1)
    print(scores)
    print("**************")
    y_hat=model.predict_classes(xtest)
    print(y_hat)
    print("**************")
    
    
    from sklearn.metrics import precision_recall_fscore_support
    print(precision_recall_fscore_support(ytest,y_hat))
    
    
main()

