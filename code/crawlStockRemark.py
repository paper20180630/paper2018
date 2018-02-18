# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:50:12 2017

@author: HTX
"""

import requests
from bs4 import BeautifulSoup
import re
import time



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
获取沪市股票代码信息getStockList
"""   
def getStockList(lst, stockURL):
    html = getHTMLText(stockURL, "GB2312")
    soup = BeautifulSoup(html, 'html.parser') 
    a = soup.find_all('a')
    for i in a:
        try:
            href = i.attrs['href']
            lst.append(re.findall(r"[s][h][6]\d{5}", href)[0])  #这里暂时只爬取沪市6开头股票          
        except:
            continue 
     
"""
获取股评信息getStockRemark，暂时只获取发帖标题信息
"""       
def getStockRemark(lst, stockURL, fpath, depth):
    count=0
    for stock in lst:
        count = count + 1
        for i in range(1,depth+1):
            url=stockURL+stock+"_"+str(i)+".html"
            time.sleep(1)
            html=getHTMLText(url)
            try:
                if html=="":
                    continue
                remarkDict={}
                soup = BeautifulSoup(html, 'html.parser')
                stockRemark= soup.find_all('div',attrs={'class':'articleh'})                
                for i in stockRemark:
                    remark=i.find_all('a')
                    if len(remark)==2:
                        remarkDict.update({'remarks': remark[0].text})
                        
                        with open(fpath+stock+'.txt', 'a', encoding='utf-8') as f:
                            f.write( str(remarkDict) + '\n' )
                            print("\r当前进度: {:.2f}%".format(count*100/len(lst)),end="")

                
            except:                
                continue
            
def main():
    stock_list_url = 'http://quote.eastmoney.com/stocklist.html'
    stock_remark_url = 'http://guba.eastmoney.com/list,'
    output_file = 'D:\crawlStockRemark\stock\\'
    slist=[]
    #print(slist)
    getStockList(slist, stock_list_url) #def getStockList(lst, stockURL):
    #print(slist)
    depth=10
    getStockRemark(slist, stock_remark_url, output_file,depth)#def getStockRemark(lst, stockURL, fpath, depth):
 
main()