# -*- coding: utf-8 -*-
"""
before this, you have to install mysql-connector-python
"""

import mysql.connector

#user information
USER_NAME = 'root'
PASSWORD = 'arai0806'
#connection
conn = mysql.connector.connect(user=USER_NAME, password=PASSWORD, host='localhost', database='enwiki')
cur = conn.cursor()

# #SQL文
# cur.execute("select page_id from page_test;")
# #pageidの配列
# pages = [ page[0] for page in cur.fetchall() ]

#sample
pages = [1,2]

for page in pages:
    #pageが属するカテゴリを取得
    cur.execute("select cl_to from categorylinks_test where cl_from="+str(page)+";")
    categories = [cl_from[0] for cl_from in cur.fetchall()]
    print(categories)

cur.close
conn.close
