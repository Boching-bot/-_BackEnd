import json
import pymssql
from flask import jsonify


def showDescribe(id, cur):
    sql = "SELECT describe FROM old_person WHERE ID='" + id + "' for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    return j


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
    sql = "DECLARE @re int EXEC register '" + id + "', '" + name + "', '" + uid + "', '" + pw + "', '" + aid + "', '" + apw + "'  select @re"
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


def insertOld_1(id, name, gender, rNo, h, cur):
    sql= "INSERT INTO old_person VALUES('"+id+"','"+name+"','"+gender+"','',GETDATE(),'','0','','','"+rNo+"','','','','"+h+"','','')"
    cur.execute(sql)
    return {'result': '0'}


def insertOld_2(id, name, tel, C1, C2, carer, cur):
    sql= "INSERT INTO old_person VALUES('"+id+"','"+name+"','','"+tel+"',GETDATE(),'','0','','','','"+C1+"','"+C2+"','','healthy','','"+carer+"')"
    cur.execute(sql)
    return {'result': '0'}


def showAllOld_1(cur):
    sql = "SELECT (SELECT COUNT(*) FROM old_person) num, name, ID, gender, roomNo, inDate, health FROM old_person for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    return j


def updateOld_1(ID, roomNo, health, cur):
    sql = "UPDATE old_person set roomNO='" + roomNo + "', health='" + health + "' WHERE ID='" + ID +"'"
    cur.execute(sql)
    return {'result': '0'}


def showAllOld_2(cur):
    sql = "SELECT (SELECT COUNT(*) FROM old_person) num, name, ID, tel, C1, C2, carer FROM old_person for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    return j


def updateOld_2(ID, tel, C1, C2, carer, cur):
    sql = "UPDATE old_person set tel='" + tel + "', C1='" + C1 + "', C2='" + C2 + "', carer='" + carer + "' WHERE ID='" + ID + "'"
    cur.execute(sql)
    return {'result': '0'}


def deleteOld(ID, cur):
    sql = "DELETE FROM old_person WHERE ID='" + ID + "'"
    cur.execute(sql)
    return {'result': '0'}


def showAllWorker(cur):
    sql = "SELECT (SELECT COUNT(*) FROM worker) num, ID, name, gender, tel, type, valid FROM worker for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    return j


def deleteWorker(ID, cur):
    sql = "DELETE FROM worker WHERE ID='" + ID + "'"
    cur.execute(sql)
    return {'result': '0'}


def updateWorker(id, tel, type, valid, cur):
    sql = "UPDATE worker SET tel='"+tel+"', type='"+type+"', valid='"+valid+"' WHERE ID='"+id+"'"
    cur.execute(sql)
    return {'result': '0'}


def insertWorker(id, name, gender, tel, type, valid, cur):
    sql = "INSERT INTO worker VALUES('"+id+"','"+name+"','"+gender+"','"+tel+"','"+type+"','"+valid+"','','')"
    cur.execute(sql)
    return {'result': '0'}


def showAllAdmin(cur):
    sql = "SELECT (SELECT COUNT(*) FROM admins) num, ID, name, gender, tel, userID, password FROM admins for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    return j


def deleteWorker(ID, cur):
    sql = "DELETE FROM admins WHERE ID='" + ID + "'"
    cur.execute(sql)
    return {'result': '0'}


def updateAdmin(id, tel, password, cur):
    sql = "UPDATE admins SET tel='"+tel+"', password='"+password+"' WHERE ID='"+id+"'"
    cur.execute(sql)
    return {'result': '0'}


def insertAdmin(id, name, gender, tel, uid, pw, cur):
    sql = "INSERT INTO admins VALUES('"+id+"', '"+name+"', '"+gender+"', '"+tel+"', '"+uid+"', '"+pw+"')"
    cur.execute(sql)
    return {'result': '0'}


def showAllCus(cur):
    sql = "SELECT (SELECT COUNT(*) FROM custodian) num, ID, name, gender, tel, relation FROM custodian for json auto"
    cur.execute(sql)
    data = cur.fetchall()
    a = list(data[0])
    j = json.loads(a[0])
    return j


def deleteCus(ID, cur):
    sql = "DELETE FROM custodian WHERE ID='" + ID + "'"
    cur.execute(sql)
    return {'result': '0'}


def updateCus(id, tel, cur):
    sql = "UPDATE custodian SET tel='"+tel+"' WHERE ID='"+id+"'"
    cur.execute(sql)
    return {'result': '0'}


def insertCus(id, name, gender, tel, re, cur):
    sql = "INSERT INTO custodian VALUES('"+id+"', '"+name+"', '"+gender+"', '"+tel+"', '"+re+"')"
    cur.execute(sql)
    return {'result': '0'}


# connect = pymssql.connect(host="LAPTOP-NJC0SCGO", user="sa", password="123456", database="ZHYL", charset="utf8",
#                            autocommit=True)
# cur = connect.cursor()
# #
# data = showAllOld_1(cur)
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
