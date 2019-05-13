import numpy as np # 科学计算库
import cv2
import cv2.face as fc
import cv2.data
import os
from PIL import Image

'''
人脸数据训练
'''

# 脸部图片路径
face_gray_img_path = 'FaceData'

recognizer = fc.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('Cascade/haarcascade_frontalface_default.xml')

# 训练数据
def getImagesAndLabels(path):
    imagePaths = [os.path.join(face_gray_img_path, f) for f in os.listdir(path)]

    faceResult = []
    ids = []

    for imagePath in imagePaths:
        # 忽略隐藏文件 .DS_Store
        if imagePath.find('.DS_Store') != -1 :
            continue

        # 读取图片，并转换为黑白图片
        '''
            convert()是图像实例对象的一个方法，接受一个 mode 参数，用以指定一种色彩模式
            1 ------------------（1位像素，黑白，每字节一个像素存储）
            L ------------------（8位像素，黑白）
            P ------------------（8位像素，使用调色板映射到任何其他模式）
            RGB------------------（3x8位像素，真彩色）
            RGBA------------------（4x8位像素，带透明度掩模的真彩色）
            CMYK--------------------（4x8位像素，分色）
            YCbCr--------------------（3x8位像素，彩色视频格式）
            I-----------------------（32位有符号整数像素）
            F------------------------（32位浮点像素）
        '''
        pil_img = Image.open(imagePath).convert('L')
        # 创建数组，数组中元素必须与第一个元素类型相同
        img_numpy = np.array(pil_img, 'uint8')
        # 获取id （样本文件中只有一个人）
        id = 0
        # 扫描图片获取数据
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceResult.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return  faceResult, ids


faces, ids = getImagesAndLabels(face_gray_img_path)
print("faces: %s" % (faces))
print("ids: %s \n np_array: %s" % (ids, np.array(ids)))
# 整合训练的数据
recognizer.train(faces, np.array(ids))
recognizer.write(r'FaceTrainer/trainer.yml')
