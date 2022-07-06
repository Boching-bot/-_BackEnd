import json
import pymssql

def login(id, pw, cur):
    sql
    return 0

def search(name, cur):
    return 0

connect = pymssql.connect(host="LAPTOP-NJC0SCGO", user="sa", password="123456", database="ZHYL", charset="utf8",
                          autocommit=True)
cur = connect.cursor()

sql = "SELECT * FROM admins for json auto"
cur.execute(sql)
data = cur.fetchall()
a = list(data[0])
j = json.loads(a[0])
print(j)