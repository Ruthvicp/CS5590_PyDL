"""
@author : ruthvicp
date : 2/7/2019
"""
import urllib.request
from bs4 import BeautifulSoup
import os
import requests

# get data on the CYP2D6 gene
url = "https://en.wikipedia.org/wiki/Deep_learning"
#html =  urllib.request.urlopen(url).read().decode('utf-8')
html = requests.get(url)
soup = BeautifulSoup(html.content, "html.parser")
# find title
title = soup.findAll('title')
print("title of the page is -------------------> ", title)

# get all <div> content and then print content for any headers
for div in soup.find_all('a'):
    aTag = div.get('href')
    # not all divs have h1 so only print if there is something to print
    if aTag != None :
        print(aTag)
# print the full text of the body
#print(soup.get_text())