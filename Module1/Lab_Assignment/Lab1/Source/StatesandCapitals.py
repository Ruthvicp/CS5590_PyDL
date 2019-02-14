import requests
from bs4 import BeautifulSoup

def collect_href():
    url = "https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India"

    source_code = requests.get(url)
    source_text = source_code.text
    soup_text = BeautifulSoup(source_text, 'html.parser')
    #print(soup_text)
    for link in soup_text.findAll('a'):
        href = link.get('href')

    table = soup_text.find("table",{"class":"wikitable sortable plainrowheaders"})

    op_file = open('output.txt', 'w')

    for rows in table.findAll('td'):
        op_file.write(rows.text)
        #print(rows.text)
    op_file.close()


if __name__ == '__main__':
    collect_href()