# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 11:00:51 2017

@author: HTX
"""

import requests

import json

headers = {'User-Agent': "Mozilla/4.0"}


"""
获取网页信息getHTMLText
"""
def getHTMLText(url, code="utf-8"):
    headers = {'User-Agent': "Mozilla/4.0"}
    try:
        r = requests.get(url,headers=headers)
        r.raise_for_status()
        r.encoding = code
        return r.text
    except:
        return ""
        
"""
获取酒店评论信息getHotlRemark
""" 
def getHotelRemark(numHotel, fpath):
    
    for i in range(numHotel):
        
        url="https://hotel.jd.com/comment/invokeComment.action?callback=comment&level=0&hotelId="+str(i)+"&pageSize=1000&curPage=1"
        html=getHTMLText(url)
        try:
            if html=="":
                continue
            jsonData=json.loads(html[8:-1])
            items=jsonData['retData']["items"]
            
            if items==[]:
                continue
            for ii in items:
                remarkDict={}
                #print(ii)
                remarkDict.update({"descr": ii['descr']})
                remarkDict.update({"star": ii['star']})
                #print(remarkDict)
                fileName=fpath+str(i)+'.txt'
               
                with open(fileName,'a',encoding='utf-8') as f:
                    
                    f.write( str(remarkDict) + '\n' )
        
                
        except:
            continue
        print("\r当前进度: {:.2f}%".format(i*100/numHotel),end="")
        
def main():
    fpath='D:/crawlHotelRemark/hotel'
    getHotelRemark(4100,fpath)
    
main()
    
            
        

    









