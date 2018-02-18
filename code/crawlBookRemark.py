# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:11:39 2017

@author: HTX
"""

import requests
from bs4 import BeautifulSoup
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
     
'''
获取豆瓣tag信息getTagList
'''
def getTagList(tagList,tagUrl):
    html=getHTMLText(tagUrl)
    soup = BeautifulSoup(html, 'html.parser')
    soup = soup.find('div',attrs={'class':'article'}).find_all('td')
    for i in soup:
        tagList.append(i.find('a').text)
    #print(tagList)

    
"""
获取书籍网页信息getBookUrlList
"""   
def getBookUrlList(list, bookTag,bookDepth):
    for i in range(bookDepth):
        bookTagUrl=bookTag+str(20*i)+"&type=R"
        time.sleep(20)
        html = getHTMLText(bookTagUrl, "GB2312")
        try:
            soup = BeautifulSoup(html, 'html.parser')
            a = soup.find_all('a',attrs={'class':'nbg'})
            if a==[]:
                continue
            for i in a:
                try:
                    href=i.attrs['href']
                    list.append(href)
                except:
                    continue       
        except:
            continue
 
                   
"""
获取书评信息getBookRemark
""" 

def getBookRemark(slist,fpath,depth):
    for bookUrl in slist:
        
        for i in range(1,depth+1):
            
            url=bookUrl+"comments/hot?p="+str(i)
            
            time.sleep(20)
            html=getHTMLText(url)
            
            try:
                if html=="":
                    continue
                remarkDict={}
                soup = BeautifulSoup(html, 'html.parser')
                bookRemark= soup.find_all('div',attrs={'class':'comment'})
                
                for i in bookRemark:
                    remarkContent=i.find('p',attrs={'class':"comment-content"})
                    
                    remarkRating=i.find('span',attrs={'class':"user-stars"})
                    try:
                        
                        remarkDict.update({'content': remarkContent.text})
                        remarkDict.update({'短评打分': remarkRating.attrs['title']})
                        with open(fpath, 'a', encoding='utf-8') as f:
                            f.write( str(remarkDict) + '\n' )
                    except:
                        continue
                                                  
            except:                
                continue

def main():
    tagList=[]
    tagUrl='https://book.douban.com/tag/?view=type&icn=index-sorttags-all'
    getTagList(tagList,tagUrl)
    print(tagList)
    count=0
    for i in tagList:
        bookDepth =5
        bookTag = 'https://book.douban.com/tag/'+i+'?start='
        output_file = 'D:/crawlBookremark/'+i+'.txt'
        slist=[]
        getBookUrlList(slist,bookTag,bookDepth)
        #print(slist)#delete
        depth=5
        getBookRemark(slist,output_file,depth)
        count=count+1
        print("\r当前进度: {:.2f}%".format(count*100/len(tagList)),end="")
 
main()
