# -*- coding: utf-8 -*-
import random
import threading
import numpy as np
from PIL import Image
import os
import time
#import pygame
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from imutils.object_detection import non_max_suppression

import cv2
import threading
from statistics import mode
import cv2
from keras.models import load_model
from torch import from_numpy, jit
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose
from action_detect.detect import action_detect
import oldcare.camera.demo as demo
import os
from math import ceil, floor


# parameters for loading data and images
detection_model_path = 'trained_models/facemodel/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/float_models/fer2013_mini_XCEPTION.33-0.65.hdf5'
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                  4: 'sad', 5: 'surprise', 6: 'neutral'}

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []
frame_window = 10


fall_net = jit.load('models/openpose1.jit')
action_net = jit.load('models/action1.jit')


# 网络模型  和  预训练模型
faceProto = 'cv2-Haar/age_gender/opencv_face_detector.pbtxt'
faceModel = 'cv2-Haar/age_gender/opencv_face_detector_uint8.pb'
ageProto = 'cv2-Haar/age_gender/age_deploy.prototxt'
ageModel = 'cv2-Haar/age_gender/age_net.caffemodel'
genderProto = 'cv2-Haar/age_gender/gender_deploy.prototxt'
genderModel = 'cv2-Haar/age_gender/gender_net.caffemodel'
# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-30)', '(38-43)', '(48-53)', '(60-100)']
ageList = ['(0-18)', '(18-32)', '(32-45)', '(45-60)', '(60-70)', '(70-80)', '(80-90)', '(90-100)']
genderList = ['Male', 'Female']
# 加载网络
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
# 人脸检测的网络和模型
faceNet = cv2.dnn.readNet(faceModel, faceProto)
padding = 20
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('TrainData/train.yml')
cascadePath = "D:\\Anaconda\\A\\envs\\flask\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
idnum = 0  # id与names数组里面的不相同，相差1


# parameters for loading data and images
detection_model_path = "D:\\Anaconda\\A\\envs\\flask\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
emotion_model_path = 'trained_models/float_models/fer2013_mini_XCEPTION.33-0.65.hdf5'
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                  4: 'sad', 5: 'surprise', 6: 'neutral'}


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
    frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
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
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),
                         8)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    return frameOpencvDnn, bboxes


def getImagesAndLabels(path):
    detector = cv2.CascadeClassifier('cv2/haarcascade_frontalface_alt.xml')
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
        'cv2/haarcascade_frontalface_alt.xml')
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
            'FaceData/User.' + str(
                face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])

    path = 'FaceData'
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        recognizer.write('TrainData/train.yml')
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
    frame2 = cv2.flip(frame, 1)
    img, bboxes = getFaceBox(faceNet, frame2)
    for bbox in bboxes:
        # 取出box框住的脸部进行检测,返回的是脸部图片
        face = frame2[max(0, bbox[1] - padding):min(bbox[3] + padding, frame2.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame2.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)  # blob输入网络进行性别的检测
        genderPreds = genderNet.forward()  # 性别检测进行前向传播
        gender = genderList[genderPreds[0].argmax()]  # 分类  返回性别类型
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        label = "{},{}".format(gender, age)
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1,
                   cv2.LINE_AA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(int(minW), int(minH)))
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # 函数cv2.face_FaceRecognizer.predict()
        # 在对一个待测人脸图像进行判断时，会寻找与当前图像距离最近的人脸图像。与哪个人脸图像最接近，就将待测图像识别为其对应的标签。
        # confiidence 0 完全匹配 <50 可以接受 >80 差别较大
        print(confidence)
        if 0 < confidence < 65:
            # idnum = names[idnum]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            idnum = "unknown"
            confidence = "{0}%".format(round(100 - confidence))
            # 保存整体图片或者只保存一个脸部图片
            face_detector = cv2.CascadeClassifier('cv2-Haar/haarcascade_frontalface_alt.xml')
            faces = face_detector.detectMultiScale(gray, 1.1, 3)
            count = 0
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
                count += 1
                # 保存图像
                cv2.imwrite('unknownData/wrong-' + str(
                    time.strftime('%Y%m%d%H%M', time.localtime(time.time()))) + '.jpg',
                            gray[y: y + h, x: x + w])
            # 发出警报声
            #file = r'D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\music\\baojing2.wav'
            # 初始化
            #pygame.mixer.init()
            # 加载音乐文件
            #track = pygame.mixer.music.load(file)
            # 开始播放音乐流
            #pygame.mixer.music.play()

            # 发送邮件提醒
            #msg_from = ''  # 发送方邮箱
            #passwd = ''  # 填入发送方邮箱的授权码（就是刚刚你拿到的那个授权码）
            #msg_to = '19301056@bjtu.edu.cn'  # 收件人邮箱
            #text_content = "有陌生人进入，危险!"  # 发送的邮件内容

            #file_path = "D:\\Python Project\\ZHYL_BackEnd\\oldcare\\camera\\unknownData\\wrong-{0}.jpg".format(
                #time.strftime('%Y%m%d%H%M', time.localtime(time.time())))

            #try:
                #send_email(msg_from, passwd, msg_to, text_content, file_path)
            #except:
                #print("邮件发送有误")
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


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def faceEmotion(frame):
    bgr_image = frame
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray_image, 1.3, 5)
    for face_coordinates in faces:
        x1, y1, width, height = face_coordinates
        x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height
        # x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        # emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_text = mode(emotion_window)
        except:
            continue
        color = (0, 0, 255)
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(rgb_image, emotion_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def fall_Detection(frame):
    height_size = 256
    cpu = False
    net = fall_net.eval()
    net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    img = frame
    orig_img = img.copy()
    heatmaps, pafs, scale, pad = demo.infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                 total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        if len(pose.getKeyPoints()) >= 10:
            current_poses.append(pose)
        # current_poses.append(pose)

    for pose in current_poses:
        pose.img_pose = pose.draw(img, show_draw=True)
        crown_proportion = pose.bbox[2] / pose.bbox[3]  # 宽高比
        pose = action_detect(action_net, pose, crown_proportion)

        if pose.pose_action == 'fall':
            pass
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 0, 255), thickness=3)
            cv2.putText(img, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        else:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            cv2.putText(img, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))

    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    #cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
    return img


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
        self.cap = cv2.VideoCapture("rtsp://admin:admin@192.168.0.145:8554/live")

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None

    def __del__(self):
        self.cap.release()

    def get_frame(self, state):
        ret, frame = self.cap.read()
        minW = 0.1 * self.cap.get(8)
        minH = 0.1 * self.cap.get(8)
        img = frame
        if ret:
            # 面部信息采集
            if state == 1:
                faceCollect(img)
            elif state == 2:
                img = cv2.flip(img, 1)
                img = unfamiliarIdenDet(img, minW, minH)
            elif state == 3:
                illegalInvasion(img)
            elif state == 4:
                img = faceEmotion(frame)
                img = fall_Detection(img)

            frame = cv2.flip(img, 1)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None

        # if ret:
        #
        #     # 1. 熟人数据采集
        #     # faceCollect(img)
        #
        #     # 2. 年龄,性别,陌生人检测
        #     #img = unfamiliarIdenDet(frame, minW, minH)
        #     #frame = img
        #
        #     # 3. 区域非法入侵检测
        #     #frame = cv2.flip(frame, 1)
        #     #illegalInvasion(frame)
        #
        #     #4. 情绪，姿态，跌倒检测
        #     frame = cv2.flip(frame, 1)
        #     frame = faceEmotion(frame)
        #     frame = fall_Detection(frame)
        #
        #     #out1 = faceEmotion(frame)
        #     #frame = fall_Detection(out1)
        #
        #     ret, jpeg = cv2.imencode('.jpg', frame)
        #     return jpeg.tobytes()
        # else:
        #     return None

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


class VideoCamera2(object):
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

    def get_frame(self,state):
        ret, frame = self.cap.read()
        minW = 0.1 * self.cap.get(3)
        minH = 0.1 * self.cap.get(4)
        img = frame
        if ret:
            # 面部信息采集
            if state == 1:
                faceCollect(img)
            elif state == 2:
                frame = unfamiliarIdenDet(img, minW, minH)
            elif state == 3:
                illegalInvasion(img)
                frame = cv2.flip(img, 1)
            elif state == 4:
                frame = cv2.flip(img, 1)
                img = faceEmotion(frame)
                frame = fall_Detection(img)


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
