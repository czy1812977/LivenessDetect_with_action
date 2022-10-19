# import the necessary packages
import sys

from imutils import face_utils
import argparse
import cv2

from torchvision import transforms
import time
import dlib
import numpy as np
from pose_liveness_video import load_model
from pose_liveness_video import pred_yaw
from Live_Detection import eyebrow_jaw_distance
from Live_Detection import nose_jaw_distance

# 构造参数并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="models/shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default=0,
                help="path to input video file")
ap.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
                default='', type=str)


# ap.add_argument("-t", "--threshold", type=float, default=0.27,
#                 help="threshold to determine closed eyes")
# ap.add_argument("-f", "--frames", type=int, default=2,
#                 help="the number of consecutive frames the eye must be below the threshold")

def dlib_play(width, Note, video_path):
    args = vars(ap.parse_args())
    # 表示脸部位置检测器
    detector = dlib.get_frontal_face_detector()
    # 表示脸部特征位置检测器
    predictor = dlib.shape_predictor(args["shape_predictor"])
    # 68个关键点的索引
    # 左右眼的索引
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # 嘴唇的索引
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    # 鼻子的索引
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    # 下巴的索引
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']
    # 左眉毛的索引
    (Eyebrow_Start, Eyebrow_End) = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
    cap = cv2.VideoCapture(video_path)
    time.sleep(1.0)
    DownFlag = -1
    UpFlag = -1
    RightFlag= -1
    LeftFlag = -1
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        (grabbed, frame) = cap.read()
        if frame is None:
            if UpFlag == 2:
                Note.write("isDown\n")
            elif UpFlag ==1:
                Note.write("isUp\n")
            # if LeftFlag ==1:
            #     Note.write("isRight\n")
            # elif LeftFlag ==2:
            #     Note.write("isLeft\n")
            else:Note.write("NonePose\n")
            break
        src_h, src_w = frame.shape[:2]
        dst_w = width
        h = int(dst_w * (float(src_h) / src_w))  # 按照ｗ做等比缩放
        frame = cv2.resize(frame, (dst_w, h))
        # frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 在灰度框中检测人脸
        rects = detector(gray, 0)
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # 提取鼻子和下巴的坐标，然后使用该坐标计算鼻子到左右脸边界的欧式距离
            nose = shape[nStart:nEnd]
            jaw = shape[jStart:jEnd]
            # 提取左眉毛的坐标，然后使用该坐标计算左眉毛到左右脸边界的欧式距离
            leftEyebrow = shape[Eyebrow_Start:Eyebrow_End]
            Eyebrow_JAW_Distance = eyebrow_jaw_distance(leftEyebrow, jaw)
            NOSE_JAW_Distance = nose_jaw_distance(nose, jaw)
            # 移植鼻子到左右脸边界的欧式距离
            face_left1 = NOSE_JAW_Distance[0]
            face_right1 = NOSE_JAW_Distance[1]
            face_left2 = NOSE_JAW_Distance[2]
            face_right2 = NOSE_JAW_Distance[3]
            # 移植左眉毛到左右脸边界的欧式距离，及左右脸边界之间的欧式距离
            eyebrow_left = Eyebrow_JAW_Distance[0]
            eyebrow_right = Eyebrow_JAW_Distance[1]
            left_right = Eyebrow_JAW_Distance[2]
            Down = left_right - eyebrow_left - eyebrow_right
            Shake1=face_right1-face_left1
            Shake2=face_right2-face_left2

            if DownFlag == -1 :
                if Down >= -3:
                    DownFlag=1
                if Down <=-12:
                    DownFlag=2
            elif DownFlag != -1 and UpFlag == -1:
                if Down <= -6 and DownFlag==1:
                    UpFlag = 2
                if Down >= -9 and DownFlag==2:
                    UpFlag =1
            if RightFlag == -1:
                if Shake1>=20 and Shake2>=20:
                    RightFlag=1
                if Shake1<=-20 and Shake2<=-20:
                    RightFlag=2
            elif RightFlag!=-1 and LeftFlag ==-1:
                if Shake1<=10 and Shake2<=10 and RightFlag==1:
                    LeftFlag=1
                if Shake1>=-10 and Shake2>=-10 and RightFlag==2:
                    LeftFlag=2
            # # 左脸大于右脸，右摇头
            # if face_left1 >= face_right1 + 2 and face_left2 >= face_right2 + 2 and flag == 0:
            #     Note.write("isRight\n")
            #     flag = 1
            # # 右脸大于左脸，左摇头
            # elif face_right1 >= face_left1 + 2 and face_right2 >= face_left2 + 2 and flag == 0:
            #     Note.write("isLeft\n")
            #     flag = 1
            # # 低头
            # elif eyebrow_left + eyebrow_right <= left_right + 3 :
            #
            #     flag = 1
            # # 抬头
            # elif eyebrow_left + eyebrow_right >= left_right - 1:
            #     Note.write("isUp\n")
            # else:
            #     Note.write("NonePose\n")
            #     flag = 1


def hope_net_play(Note, video_path,width):
    model = load_model()
    print("加载detector")
    # detector=dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
    detector = dlib.get_frontal_face_detector()
    # 0为默认 即摄像头，也可输入路径
    print("加载成功")
    cap = cv2.VideoCapture(video_path)
    print("抓取成功")
    transformations = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    DownFlag = -1
    UpFlag = -1
    RightFlag = -1
    LeftFlag = -1
    while True:
        _, frame = cap.read()
        if frame is None:
            if UpFlag == 2:
                Note.write("isDown\n")
            elif UpFlag == 1:
                Note.write("isUp\n")
            else:
                Note.write("NonePose\n")
            # if LeftFlag ==1:
            #     Note.write("isRight\n")
            # elif LeftFlag ==2:
            #     Note.write("isLeft\n")
            # else:Note.write("NonePose\n")
            break
        src_h, src_w = frame.shape[:2]
        dst_w = width
        h = int(dst_w * (float(src_h) / src_w))  # 按照ｗ做等比缩放
        frame = cv2.resize(frame, (dst_w, h))
        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(cv2_frame,1)
        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.left()
            y_min = det.top()
            x_max = det.right()
            y_max = det.bottom()
            # x_min = det.rect.left()
            # y_min = det.rect.top()
            # x_max = det.rect.right()
            # y_max = det.rect.bottom()
            # conf = det.confidence
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
                    # Crop image
                    # 必须强制类型转化不然就会报错
            img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
            yaw, pitch = pred_yaw(model, img, transformations)
            Down = pitch.data.cpu().numpy().tolist()
            Shake =yaw.data.cpu().numpy().tolist()
            if DownFlag == -1:
                if Down <= -25:
                    DownFlag = 1
                if Down >= 10:
                    DownFlag = 2
            elif DownFlag != -1 and UpFlag == -1:
                if Down >=-10 and DownFlag == 1:
                    UpFlag = 2
                if Down <= 5 and DownFlag == 2:
                    UpFlag = 1
            if RightFlag == -1:
                if Shake >=30:
                    RightFlag = 1
                if Shake <=-30:
                    RightFlag = 2
            elif RightFlag != -1 and LeftFlag == -1:
                if Shake <=15 and RightFlag == 1:
                    LeftFlag = 2
                if Shake >=-15 and RightFlag == 2:
                    LeftFlag = 1
                # if pitch.data.cpu().numpy().tolist() >= -10:
                #     Note.write("isUp\n")
                #
                # elif pitch.data.cpu().numpy().tolist() <= -30:
                #     Note.write("isDown\n")
                #
                # elif yaw.data.cpu().numpy().tolist() <= -30:
                #     Note.write("isLeft\n")
                #
                # elif yaw.data.cpu().numpy().tolist() >= 30:
                #     Note.write("isRight\n")
                # else:
                #     Note.write("NonePose\n")
                # if img is None:
                #     break
        if cv2.waitKey(10) == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    note1 = open('result/dlib_result.txt', mode='w')
    note2 = open('result/hope_net_result.txt', mode='w')
    base_path = 'test_video/onlyhead/down/'
    for i in range(4):
        path = base_path + str(i + 1) + ".mp4"
        print(path)
        dlib_play(600, note1, path)
        hope_net_play(note2, path,600)


if __name__ == '__main__':
    main()
