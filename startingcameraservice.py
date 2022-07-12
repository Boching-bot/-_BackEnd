# -*- coding: utf-8 -*-

'''
启动摄像头主程序

用法:
python startingcameraservice.py
python startingcameraservice.py --location room

直接执行即可启动摄像头，浏览器访问 http://192.168.1.156:5001/ 即可看到
摄像头实时画面

'''
import argparse
from flask import Flask, render_template, Response, request, jsonify
from oldcare.camera import VideoCamera
import database.utils as util
import json
import pymssql
from flask_cors import  CORS

# 传入参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--location", required=False,
                default='room', help="")
args = vars(ap.parse_args())
location = args['location']



if location not in ['room', 'yard', 'corridor', 'desk']:
    raise ValueError('location must be one of room, yard, corridor or desk')

# API
app = Flask(__name__)
CORS(app, resources=r'/*')

connect = pymssql.connect(host="LAPTOP-NJC0SCGO", user="sa", password="123456", database="ZHYL", charset="utf8",
                          autocommit=True)
cur = connect.cursor()

video_camera = None
global_frame = None


@ app.route('/')

@app.route('/login', methods=['GET'])
def login():
    id = request.args.to_dict().get('id')
    pw = request.args.to_dict().get('pw')
    result = util.login(id, pw, cur)
    return jsonify(result)


@app.route('/register', methods=['GET'])
def register():
    id = request.args.to_dict().get('id')
    name = request.args.to_dict().get('name')
    uid = request.args.to_dict().get('uid')
    pw = request.args.to_dict().get('pw')
    aid = request.args.to_dict().get('aid')
    apw = request.args.to_dict().get('apw')
    result = util.register(id, name, uid, pw, aid, apw, cur)
    return jsonify(result)


@app.route('/showAllOld_1', methods=['GET'])
def showAllOld_1():
    result = util.showAllOld_1(cur)
    return jsonify(result)


@app.route('/showAllOld_2', methods=['GET'])
def showAllOld_2():
    result = util.showAllOld_2(cur)
    return jsonify(result)


@app.route('/deleteOld', methods=['GET'])
def deleteOld():
    id = request.args.to_dict().get('ID')
    result = util.deleteOld(id, cur)
    return jsonify(result)


@app.route('/updateOld_1', methods=['GET'])
def updateOld_1():
    id = request.args.to_dict().get('ID')
    rNo = request.args.to_dict().get('roomNo')
    h = request.args.to_dict().get('health')
    result = util.updateOld_1(id, rNo, h, cur)
    return jsonify(result)


@app.route('/updateOld_2', methods=['GET'])
def updateOld_2():
    id = request.args.to_dict().get('ID')
    tel = request.args.to_dict().get('tel')
    C1 = request.args.to_dict().get('Cone')
    # if not C1:
    #     C1 = ''
    C2 = request.args.to_dict().get('Ctwo')
    # if not C2:
    #     C2 = ''
    carer = request.args.to_dict().get('carer')
    #print(id, tel, C1, C2, carer)
    result = util.updateOld_2(id, tel, C1, C2, carer, cur)
    return jsonify(result)


@app.route('/insertOld_1', methods=['GET'])
def insertOld_1():
    id = request.args.to_dict().get('ID')
    name = request.args.to_dict().get('name')
    gender = request.args.to_dict().get('gender')
    rNo = request.args.to_dict().get('roomNo')
    h = request.args.to_dict().get('health')
    result = util.insertOld_1(id,name,gender,rNo,h,cur)
    return jsonify(result)


@app.route('/insertOld_2', methods=['GET'])
def insertOld_2():
    id = request.args.to_dict().get('ID')
    name = request.args.to_dict().get('name')
    tel = request.args.to_dict().get('tel')
    C1 = request.args.to_dict().get('Cone')
    C2 = request.args.to_dict().get('Ctwo')
    carer = request.args.to_dict().get('carer')
    result = util.insertOld_2(id, name, tel, C1, C2, carer, cur)
    return jsonify(result)


@app.route('/showAllWorker', methods=['GET'])
def showAllWorker():
    result = util.showAllWorker(cur)
    return jsonify(result)


@app.route('/deleteWorker', methods=['GET'])
def deleteWorker():
    id = request.args.to_dict().get('ID')
    result = util.deleteWorker(id,cur)
    return jsonify(result)


@app.route('/updateWorker', methods=['GET'])
def updateWorker():
    id = request.args.to_dict().get('ID')
    tel = request.args.to_dict().get('tel')
    type = request.args.to_dict().get('type')
    valid = request.args.to_dict().get('valid')
    #print(id,tel,type,valid)
    result = util.updateWorker(id, tel, type, valid, cur)
    return jsonify(result)


@app.route('/insertWorker', methods=['GET'])
def insertWorker():
    id = request.args.to_dict().get('ID')
    name = request.args.to_dict().get('name')
    gender = request.args.to_dict().get('gender')
    tel = request.args.to_dict().get('tel')
    t = request.args.to_dict().get('type')
    v = request.args.to_dict().get('valid')
    result = util.insertWorker(id, name, gender, tel, t, v, cur)
    return jsonify(result)

@app.route('/showAllAdmin', methods=['GET'])
def showAllAdmin():
    result = util.showAllAdmin(cur)
    return jsonify(result)


@app.route('/deleteAdmin', methods=['GET'])
def deleteAdmin():
    id = request.args.to_dict().get('ID')
    result = util.deleteWorker(id,cur)
    return jsonify(result)


@app.route('/updateAdmin', methods=['GET'])
def updateAdmin():
    id = request.args.to_dict().get('ID')
    tel = request.args.to_dict().get('tel')
    password = request.args.to_dict().get('password')
    result = util.updateAdmin(id, tel, password, cur)
    return jsonify(result)


@app.route('/insertAdmin', methods=['GET'])
def insertAdmin():
    id = request.args.to_dict().get('ID')
    name = request.args.to_dict().get('name')
    gender = request.args.to_dict().get('gender')
    tel = request.args.to_dict().get('tel')
    uid = request.args.to_dict().get('userID')
    pw = request.args.to_dict().get('password')
    result = util.insertAdmin(id, name, gender, tel, uid, pw, cur)
    return jsonify(result)



def index():
    return render_template(location + '_camera.html')


@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera
    if video_camera == None:
        video_camera = VideoCamera()

    status = request.form.get('status')
    save_video_path = request.form.get('save_video_path')

    if status == "true":
        video_camera.start_record(save_video_path)
        return 'start record'
    else:
        video_camera.stop_record()
        return 'stop record'


def video_stream():
    global video_camera
    global global_frame

    if video_camera is None:
        video_camera = VideoCamera()

    while True:
        frame = video_camera.get_frame()

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame
                   + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + global_frame + b'\r\n\r\n')


@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
        app.run(host='0.0.0.0', threaded=True, port=5001)
