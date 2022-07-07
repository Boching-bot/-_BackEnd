# -*- coding: utf-8 -*-
import random
import threading
import cv2
import numpy as np
from PIL import Image
import os
import cv2 as cv
import time
import pygame
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from imutils.object_detection import non_max_suppression

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
    frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()  # 网络进行前向传播，检测人脸
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])  # bounding box 的坐标
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),
                         8)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    return frameOpencvDnn, bboxes


# 网络模型  和  预训练模型
faceProto = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\age_gender\\opencv_face_detector.pbtxt"
faceModel = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\age_gender\\opencv_face_detector_uint8.pb"
ageProto = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\age_gender\\age_deploy.prototxt"
ageModel = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\age_gender\\age_net.caffemodel"
genderProto = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\age_gender\\gender_deploy.prototxt"
genderModel = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\age_gender\\gender_net.caffemodel"
# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-30)', '(38-43)', '(48-53)', '(60-100)']
ageList = ['(0-18)', '(18-32)', '(32-45)', '(45-60)', '(60-70)', '(70-80)', '(80-90)', '(90-100)']
genderList = ['Male', 'Female']
# 加载网络
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
# 人脸检测的网络和模型
faceNet = cv.dnn.readNet(faceModel, faceProto)
padding = 20
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\TrainData\\train.yml')
cascadePath = r'D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\cv2-Haar\\haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
idnum = 0  # id与names数组里面的不相同，相差1


# parameters for loading data and images
detection_model_path = 'trained_models/facemodel/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/float_models/fer2013_mini_XCEPTION.33-0.65.hdf5'
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                  4: 'sad', 5: 'surprise', 6: 'neutral'}

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def getImagesAndLabels(path):
    detector = cv2.CascadeClassifier('d:\\python3.9\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')  # 图片格式转换
        if os.path.split(imagePath)[-1].split(".")[-1] != 'jpg':
            continue
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)  # 人脸检测
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids


def faceCollect(img):
    face_detector = cv2.CascadeClassifier(
        'd:\\python3.9\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')
    face_id = "1"
    # 转为灰度图片
    count = random.randint(1,20)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
        # 保存图像
        cv2.imwrite(
            "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\FaceData\\User." + str(
                face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])

    path = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\FaceData"
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        recognizer.write(r'D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\TrainData\train.yml')
    except:
        print()


def send_email(msg_from, passwd, msg_to, text_content, file_path=None):
    msg = MIMEMultipart()
    subject = "陌生人进入，危险报警 ！"  # 主题
    text = MIMEText(text_content)
    msg.attach(text)
    # docFile = 'C:/Users/main.py'  如果需要添加附件，就给定路径
    if file_path:  # 最开始的函数参数我默认设置了None ，想添加附件，自行更改一下就好
        docFile = file_path
        docApart = MIMEApplication(open(docFile, 'rb').read())
        docApart.add_header('Content-Disposition', 'attachment', filename=docFile)
        msg.attach(docApart)
        print('发送附件！')
    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to
    try:
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print("发送成功")
    except smtplib.SMTPException as e:
        print("发送失败")
    finally:
        s.quit()


def unfamiliarIdenDet(frame,minW,minH):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # ret, img = cam.read()
    t = time.time()
    # hasFrame, frame = cam.read()
    frame2 = cv.flip(frame, 1)
    img, bboxes = getFaceBox(faceNet, frame2)
    for bbox in bboxes:
        # 取出box框住的脸部进行检测,返回的是脸部图片
        face = frame2[max(0, bbox[1] - padding):min(bbox[3] + padding, frame2.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame2.shape[1] - 1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)  # blob输入网络进行性别的检测
        genderPreds = genderNet.forward()  # 性别检测进行前向传播
        gender = genderList[genderPreds[0].argmax()]  # 分类  返回性别类型
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        label = "{},{}".format(gender, age)
        cv.putText(img, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1,
                   cv.LINE_AA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(int(minW), int(minH)))
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print(confidence)
        if 0 < confidence < 65:
            # idnum = names[idnum]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            idnum = "unknown"
            confidence = "{0}%".format(round(100 - confidence))
            # 保存整体图片或者只保存一个脸部图片
            face_detector = cv2.CascadeClassifier(r'd:\\python3.9\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')
            faces = face_detector.detectMultiScale(gray, 1.1, 3)
            count = 0
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
                count += 1
                # 保存图像
                cv2.imwrite("D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\unknownData\\wrong-" + str(
                    time.strftime('%Y%m%d%H%M', time.localtime(time.time()))) + '.jpg',
                            gray[y: y + h, x: x + w])
            # 发出警报声
            file = r'D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\music\\baojing2.wav'
            # 初始化
            pygame.mixer.init()
            # 加载音乐文件
            track = pygame.mixer.music.load(file)
            # 开始播放音乐流
            pygame.mixer.music.play()

            # 发送邮件提醒
            msg_from = ''  # 发送方邮箱
            passwd = ''  # 填入发送方邮箱的授权码（就是刚刚你拿到的那个授权码）
            msg_to = '19301056@bjtu.edu.cn'  # 收件人邮箱
            text_content = "有陌生人进入，危险!"  # 发送的邮件内容

            file_path = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\unknownData\\wrong-{0}.jpg".format(
                time.strftime('%Y%m%d%H%M', time.localtime(time.time())))

            try:
                send_email(msg_from, passwd, msg_to, text_content, file_path)
            except:
                print("邮件发送有误")
    try:
        cv2.putText(img, str(idnum), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 0), 1)
    except:
        print()
    return img

def illegalInvasion(img):
    (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
    # 设置来抑制重叠的框
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # 绘制红色人体矩形框
    for (x, y, w, h) in pick:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if len(pick) > 0:
        print("检测到进入危险区域物体个数为{}".format(len(pick)))


class RecordingThread(threading.Thread):
    def __init__(self, name, camera, save_video_path):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # MJPG
        self.out = cv2.VideoWriter(save_video_path, fourcc, 20.0,
                                   (640, 480), True)

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()


class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None

    def __del__(self):
        self.cap.release()

    def get_frame(self):

        ret, frame = self.cap.read()

        minW = 0.1 * self.cap.get(3)
        minH = 0.1 * self.cap.get(4)

        img = frame

        if ret:

            # 1. 熟人数据采集
            # faceCollect(img)

            # 2. 年龄,性别,陌生人检测
            # img = unfamiliarIdenDet(frame,minW,minH)

            # 3. 区域非法入侵检测
            # illegalInvasion(img)

            frame = cv2.flip(img, 1)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None

    def start_record(self, save_video_path):
        self.is_record = True
        self.recordingThread = RecordingThread(
            "Video Recording Thread",
            self.cap, save_video_path)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()
