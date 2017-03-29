# coding: utf-8

import urllib
import urllib.parse
import urllib.request
# import urllib2
from xml.dom.minidom import parse as parseXML

from gensim.corpora import WikiCorpus

from gensim.corpora.wikicorpus import filter_wiki

import re

#mysql
import mysql.connector

#user information
USER_NAME = 'root'
PASSWORD = 'arai0806'
#connection
conn = mysql.connector.connect(user=USER_NAME, password=PASSWORD, host='localhost', database='enwiki')
cur = conn.cursor()


URL = 'http://en.wikipedia.org/w/api.php?'
BASIC_PARAMETERS = {'action': 'query',
                    'format': 'xml'}

class WikiHandler(object):
    def __init__(self, parameters, titles=None, url=URL):
        self._url = url if url.endswith('?') else url + '?'

        self._parameters = {}
        self._parameters.update(BASIC_PARAMETERS)
        self._parameters.update(parameters)

        if titles:
            self._parameters['titles'] = titles

        self.rawdata = self._urlfetch(self._parameters)

        if self._parameters['format'] == 'xml':
            self.dom = parseXML(self.rawdata)
            print('DOM ready.')

    def _urlfetch(self, parameters):
        parameters_list = []

        for key, val in parameters.items():
            if isinstance(val, str):
                val = val.encode('utf-8')
            else:
                val = str(val)

            val = urllib.parse.quote(val)
            parameters_list.append('='.join([key, val]))

        url = self._url + '&'.join(parameters_list)

        print('Accessing...\n', url)

        return urllib.request.urlopen(url, timeout=20)

def category_tracing(lists):
    result = []
    temp_lists = lists
    for index, l in enumerate(lists):
        if 'Category:' in l:
            # print(l)
            temp_lists.pop(index)
            parameters = {'list': 'categorymembers',
                          'cmlimit': 500,
                          'cmtitle': l}
            page = WikiHandler(parameters)
            elelist = page.dom.getElementsByTagName('cm')
            for ele in elelist:
                # print(ele.getAttribute('title').encode('sjis', 'ignore').decode('sjis', 'ignore'))
                result.append(ele.getAttribute('title').encode('sjis', 'ignore').decode('sjis', 'ignore'))

    return temp_lists + result

    def get_text(self):
        result = ""
        elelist = self.dom.getElementsByTagName('rev')
        if elelist.length is not 0:
            ele = elelist[0]
            s = filter_wiki(ele.childNodes[0].data).encode('sjis', 'ignore')
            result = re.sub(r'[^a-zA-Z ]', '', s.decode('sjis', 'ignore')).lower()
        return result

def get_text_by_pageids(pageids):
    parameters = {'prop': 'revisions',
                  'rvprop': 'content',
                  'pageids': pageids}
    page = WikiHandler(parameters)
    return page.get_text()

if __name__ == '__main__':
    result = []
    article_num = 100

    #SQL文
    cur.execute("select page_id from page_test;")
    #pageidの配列
    pages = [ page[0] for page in cur.fetchall() ]

    for page in pages[:article_num]:
        if len(get_text_by_pageids(page)) is not 0:
            sentence = get_text_by_pageids(page)
            cur.execute("select cl_to from categorylinks_test where cl_from="+str(page)+";")
            categories = [cl_from[0] for cl_from in cur.fetchall()]
            result.append((categories, sentence))

    #データの書き込み
    with open('test.pickle', 'wb') as f:
        pickle.dump(result, f)
