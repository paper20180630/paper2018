# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:32:06 2018

@author: HTX
"""

import pandas as pd
import numpy as np
import os


'''
提取京东酒店评论数据并处理 getHotelRemarkData(hotelPath)
hotelPath为豆瓣评价文件存放位置
'''
def getHotelRemarkData(hotelPath,samNum):
    content=[]
    rating=[]
    length=[]
    files=os.listdir(hotelPath)
    for file in files:
        if not os.path.isdir(file):
            hotelRemarkData=open(hotelPath+"/"+file,encoding= 'utf-8')
            for i in hotelRemarkData:
                
                try:
                    rating.append(i.split("\', \'")[1][7:-2])
                    content.append(i.split("\', \'")[0][10:])                    
                    length.append(len(i.split("\', \'")[0][10:]))
                except:
                    continue
    
    hotelRemarkAll=pd.DataFrame({"content":content,"rating":rating,"length":length})
    hotelRemarkAll=hotelRemarkAll[hotelRemarkAll["length"]<25]
    pos=hotelRemarkAll[hotelRemarkAll["rating"]=='5']
    pos=pos.sample(n=samNum)
    
    neg=hotelRemarkAll[(hotelRemarkAll["rating"]=='1')|(hotelRemarkAll["rating"]=='2')]
    neg=neg.sample(n=samNum)                  
    
    pos['mark']=1

    neg['mark']=0

    hotelRemarkPd=pos.append(neg).drop(['rating'],axis=1)
    return hotelRemarkPd
    
'''
提取豆瓣评论数据并处理 getBookRemarkData(bookPath) 
bookpath为豆瓣评价文件存放位置
'''
def getBookRemarkData(bookPath,samNum):
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
    pos=pos.sample(n=samNum)
    neg=bookRemarkAll[(bookRemarkAll["rating"]=='很差')|(bookRemarkAll["rating"]=='较差')]
    neg=neg.sample(n=samNum)
    pos['mark']=1
    neg['mark']=0
    bookRemarkPd=pos.append(neg).drop(['rating'],axis=1)
    return bookRemarkPd

    

'''
One Hot编码处理 getOneHot(remarkPd,maxLen,minCount,dic)
'''

def makeDic(remarkPd,maxLen,minCount):
    remarkAll=remarkPd
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
    
    
def main():
    samNum=51423
    print(type(samNum))
    hotelPath="D:/crawlHotelremark"
    hotelRemarkPd=getHotelRemarkData(hotelPath,samNum)
    bookPath="C:/Users/tianxiang-hu/Desktop/HTX/HTX/数据集/标记数据集/豆瓣带标记数据集"
    bookRemarkPd=getBookRemarkData(bookPath,samNum)
    remarkPd=hotelRemarkPd.append(bookRemarkPd, ignore_index=True)
    minCount=30
    maxLen=25
    dic=makeDic(remarkPd,maxLen,minCount)
    dic.to_hdf('dic.h5','df')
    hotelRemarkPd=getOneHot(hotelRemarkPd,maxLen,minCount,dic)
    hotelRemarkPd.to_hdf('hotelRemarkPd.h5','df')
    bookRemarkPd=getOneHot(bookRemarkPd,maxLen,minCount,dic)
    bookRemarkPd.to_hdf('bookRemarkPd','df')
    
    
    from sklearn.utils import shuffle
    bookRemarkPd=shuffle(bookRemarkPd)
    hotelRemarkPd=shuffle(hotelRemarkPd)
  
    xbook=np.array(list(bookRemarkPd['sent2Num']))
    xtrainbook=xbook[0:int(len(xbook)*0.8)]
    xtestbook=xbook[int(len(xbook)*0.8):]
   
    ybook=np.array(list(bookRemarkPd['mark']))
    ytrainbook=ybook[0:int(len(xbook)*0.8)]
    ytestbook=ybook[int(len(xbook)*0.8):]
    
    xhotel=np.array(list(hotelRemarkPd['sent2Num']))
    xtrainhotel=xhotel[0:int(len(xhotel)*0.8)]
    xtesthotel=xhotel[int(len(xhotel)*0.8):]
   
    yhotel=np.array(list(hotelRemarkPd['mark']))
    ytrainhotel=yhotel[0:int(len(xhotel)*0.8)]
    ytesthotel=yhotel[int(len(xhotel)*0.8):]
    
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
    inputBook = Input(shape=(25,), dtype='int32', name='inputBook')
    inputHotel = Input(shape=(25,), dtype='int32', name='inputHotel')
    outBook=nlpModel(inputBook)
    outHotel=nlpModel(inputHotel)
    denseBook1=Dense(32,activation='relu',name='densebook1')(outBook)
    denseBook2=Dense(1,activation='sigmoid',name='densebook2')(denseBook1)
    denseHotel1=Dense(32,activation='relu',name='densehotel1')(outHotel)
    denseHotel2=Dense(1,activation='sigmoid',name='densehotel2')(denseHotel1)
    model = Model(inputs=[inputBook, inputHotel],outputs=[denseBook2,denseHotel2])
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'],loss_weights=[0.5, 0.5])

    model.fit([xtrainbook,xtrainhotel],[ytrainbook,ytrainhotel],epochs=10,batch_size=256)
    
    print("**************")
    
    print(model.evaluate([xtestbook,xtesthotel],[ytestbook,ytesthotel],verbose=1))
    print(model.metrics_names)
    [ybook_hat,yhotel_hat]=model.predict([xtestbook,xtesthotel])
    ybook_hat=np.around(ybook_hat)
    yhotel_hat=np.around(yhotel_hat)

    model.save("mymodel.h5")
    model.save_weights('mymodelweight.h5')      

    
    from sklearn.metrics import precision_recall_fscore_support
    print(precision_recall_fscore_support(ybook_hat,ytestbook))
    print(precision_recall_fscore_support(yhotel_hat,ytesthotel))
    
    
main()
    