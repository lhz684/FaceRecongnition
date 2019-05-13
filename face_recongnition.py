import cv2
import cv2.face as fc

recongnizer = fc.LBPHFaceRecognizer_create()
recongnizer.read('FaceTrainer/trainer.yml')
cascadePath = 'Cascade/haarcascade_frontalface_default.xml'
faceCascades = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

id = 0

names = ['lhz', '哈哈哈']

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascades.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:


        # 比对数据
        id, confidence = recongnizer.predict(gray[y:y+h, x:x+w])

        color = (0,0,0)
        # 概率 = 100 - 置信度
        if confidence < 100:
            idnum = names[id]
            confidence = '{0}%'.format(round(100 - confidence))
            color = (0,255,0)
        else:
            idnum = 'unknown'
            confidence = '{0}%'.format(round(100 - confidence))
            color = (0, 0, 255)
        # 框选人脸
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # 显示label
        cv2.putText(img, str(idnum), (x, y), font, 1, color, 1)
        # 显示精度
        cv2.putText(img, str(confidence), (x, y + h - 5), font, 1, color, 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()