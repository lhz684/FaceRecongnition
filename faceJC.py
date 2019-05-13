
import numpy as np
import cv2

'''
    人脸检测
'''

# 脸部识别器
faceCascade = cv2.CascadeClassifier(r'Cascade/haarcascade_frontalface_default.xml')

# 眼部识别器
eyesCascade = cv2.CascadeClassifier(r'Cascade/haarcascade_eye_tree_eyeglasses.xml')

# 开启摄像头
cap = cv2.VideoCapture(0)
ok = True

face_count = 0

while ok:
    # 读取摄像头图像， ok为是否读取成功
    ok, img = cap.read()
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人类检测
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors = 5,
        minSize=(32,32)
    )

    # 在人类检测的基础上检测眼睛
    result = []
    for (x, y, w, h) in faces:

        face_gray = gray[y:(y+h), x:(x+w)]

        #TODO 获取样本

        if face_count < 10:
            # 保存样本图片
            cv2.imwrite('FaceData/lhz.' + str(face_count) + '.jpg', face_gray)
            face_count += 1

        eyes = eyesCascade.detectMultiScale(face_gray, 1.3, 2)
        # 眼睛坐标换算，将相对位置换为绝对位置
        for (ex, ey, ew, eh) in eyes:
            result.append((x+ex, y + ey, ew, eh))

    # 画矩形
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

    for (ex, ey, ew, eh) in result:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('video', img)


    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()