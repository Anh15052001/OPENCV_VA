"""
Có 2 cách thực hiện:
C1:Phát hiện khuôn mặt bằng SSD, YOLO sau đó predict ưu điểm độ chính xác cao nhưng không có data để train
C2: Phát hiện khuôn mặt haarcascade, capture cùng miệng(landmask), tính average saturation, so sánh với
1 ngưỡng threshold ưu điểm: không cần data, tốc độ nhanh nhưng độ chính xác thấp khi điều kiện ánh sáng thay đổi
imutils: hỗ trợ opencv, resize, rotation
dlib: hỗ trợ landmask, vẽ rectangle
Các bước:
Khởi tạo bộ haarcascade, landmask, videoStream -> Đọc ảnh liên tục từ cam
(resize ảnh, chuyển về ảnh xám detectMultiScale) -> duyệt qua các mặt
vẽ hình chữ nhật, nhận diện các điểm landmask -> capture vùng miệng-> tính average saturaton
(hsv) so sánh với 1 threshold
"""
from imutils.video import VideoStream
from imutils import face_utils #capture vùng miệng
import imutils
import numpy as np
import cv2
import dlib

import pygame
import time
#Bước 1: Khởi tạo các bộ haarcascade, landmask, VideoStream
#bộ phát hiện khuôn mặt
face_detect =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#bộ landmask
landmask_detect=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#khởi tạo camera VideoStream
vs=VideoStream(src=0).start()
time.sleep(1.0)
#bước 2: Đọc ảnh liên tục từ cam
#resize kích thước ảnh (imutils) chuyển về ảnh xám phục vụ phát hiện
#khuôn mặt detectMultiScale
while True:
    frame=vs.read()
    #resize ảnh để tăng tốc độ xử lí
    frame=imutils.resize(frame, width=600)
    #chuyển về ảnh xám phục vụ detectMultiScale
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Phát hiện các khuôn mặt có trong hình nhờ detectMultiscale
    faces=face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    #bước 3: duyệt qua lần lượt từng mặt
    #vẽ rectangle và nhận diện các điểm landmask
    for (x, y, w, h) in faces:
        #tạo 1 hcn quanh mặt
        rect=dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        #nhận diện các điểm landmask
        landmask=landmask_detect(gray, rect)
        #chuyển về numpy
        landmask=face_utils.shape_to_np(landmask)
        #capture vùng miệng
        (mStart, mEnd)=face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        mouth=landmask[mStart:mEnd]
        #Lấy hcn bao quanh vùng miêngj
        boundRect=cv2.boundingRect(mouth)
        #các bước từ detectMulti, nhận diện các điểm landmask
        #là làm việc với ảnh xám
        #bây giờ ta sẽ làm việc với ảnh màu và chuyển về HSV
        cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])),
                      (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0, 0, 255), 2)
        #bước 4: tính saturation trung bình và so sánh
        hsv=cv2.cvtColor(frame[int(boundRect[1]):int(boundRect[1]+boundRect[3]),
                         int(boundRect[0]):int(boundRect[0]+boundRect[2])], cv2.COLOR_BGR2HSV)
        sum_saturation=np.sum(hsv[:, :, 1])
        #area
        area=int(boundRect[2]*boundRect[3])
        avg_saturation=sum_saturation/area
        #kiểm tra với 1 ngưỡng threshold
        pygame.init()
        if avg_saturation > 100:
            cv2.putText(frame, "CHECK KHAU TRANG", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            pygame.mixer.music.load('khautrang.mp3')
            pygame.mixer.music.play()

        else:
            cv2.putText(frame, "BAN GIU AN TOAN TOT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            pygame.mixer.music.pause()

    #hiển thị lên màn hình
    cv2.imshow("Camera", frame)
    #bấm esc thoát
    key=cv2.waitKey(1)&0xFF
    if key==27:
        break
cv2.destroyAllWindows()
vs.stop()