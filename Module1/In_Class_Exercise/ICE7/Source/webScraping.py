"""
@author : ruthvicp
date : 2/7/2019
"""
import urllib.request
from bs4 import BeautifulSoup
import os
import requests


url = "https://en.wikipedia.org/wiki/Google"
#html =  urllib.request.urlopen(url).read().decode('utf-8')
html = requests.get(url)
soup = BeautifulSoup(html.content, "html.parser")
# find title
title = soup.find('div', {'class':'mw-content-ltr'})
with open('input.txt', 'w') as fp:
    fp.write( title.text)
