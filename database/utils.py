import json
import pymssql
from flask import jsonify


def login(id, pw, cur):
    sql = "SELECT name FROM admins WHERE userID='" + id + "' AND password='" + pw + "'"
    cur.execute(sql)
    data = cur.fetchall()
    #return data
    if not data:
        print('Wrong')
        return [{'result': 'Wrong'}]
    else:
        print(data[0][0])
        return [{'result': '1', 'id': data[0][0]}]


def register(id, name, uid, pw, aid, apw, cur):
    sql = "EXEC register '" + id + "', '" + name + "', '" + uid + "', '" + pw + "', '" + aid + "', '" + apw + "'"
    cur.execute(sql)
    data = cur.fetchall()
    return data


def search(name, cur):
    sql = "SELECT * FROM old_people WHERE name='" + name + "' for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    #return jsonify(j)
    return j



connect = pymssql.connect(host="LAPTOP-NJC0SCGO", user="sa", password="123456", database="ZHYL", charset="utf8",
                          autocommit=True)
cur = connect.cursor()

data = login('123456', '123456', cur)
print(data)
#sql = "SELECT * FROM admins for json auto"
#cur.execute(sql)
#data = cur.fetchall()
# if(data):
#     a = list(data[0])
#     j = json.loads(a[0])
#     print(a[0])
#     print(j)
# else:
#     print("none")
