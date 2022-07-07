import json
import pymssql


def login(id, pw, cur):
    sql = "SELECT * FROM admins WHERE userID='" + id + "' AND password='" + pw + "' for json auto"
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

data = login('123456','123456',cur)
#sql = "SELECT * FROM admins for json auto"
#cur.execute(sql)
#data = cur.fetchall()
if(data):
    a = list(data[0])
    j = json.loads(a[0])
    print(j)
else:
    print("none")
