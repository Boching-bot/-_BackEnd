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
        return [{'result': '1'}]
    else:
        print(data[0][0])
        return {'result': '0', 'name': data[0][0]}


def register(id, name, uid, pw, aid, apw, cur):
    sql = "EXEC register '" + id + "', '" + name + "', '" + uid + "', '" + pw + "', '" + aid + "', '" + apw + "'"
    cur.execute(sql)
    data = cur.fetchall()
    return {'result': data}


def search(name, cur):
    sql = "SELECT * FROM old_people WHERE name='" + name + "' for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    #return jsonify(j)
    return j


def showAllOld_1(cur):
    sql = "SELECT name, ID, gender, roomNo, inDate, health FROM old_people for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    return j


def updateOld_1(ID, roomNo, health, cur):
    sql = "UPDATE old_people set roomNO='" + roomNo + "', health='"  + health + "', WHERE ID='" + ID +"'"
    cur.execute(sql)
    return {'result': '0'}


def showAllOld_2(cur):
    sql = "SELECT name, ID, tel, C1, C2, carer FROM old_people for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    return j


def updateOld_2(ID, tel, C1, C2, carer, cur):
    sql = "UPDATE old_people set tel='" + tel + "', C1='" + C1 + "', C2='" + C2 + "', carer='" + carer + "', WHERE ID='" + ID + "'"
    cur.execute(sql)
    return {'result': '0'}


def deleteOld(ID, cur):
    sql = "DELETE FROM old_people WHERE ID='" + ID + "'"
    cur.execute(sql)
    return {'result': '0'}

# connect = pymssql.connect(host="LAPTOP-NJC0SCGO", user="sa", password="123456", database="ZHYL", charset="utf8",
#                           autocommit=True)
# cur = connect.cursor()
#
# data = login('123456', '123456', cur)
# print(data)
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
