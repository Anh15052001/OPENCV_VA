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
import streamlit as st
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
def OpenCV():
    # Bước 1: Khởi tạo các bộ haarcascade, landmask, VideoStream
    # bộ phát hiện khuôn mặt
    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # bộ landmask
    landmask_detect = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # khởi tạo camera VideoStream
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    # bước 2: Đọc ảnh liên tục từ cam
    # resize kích thước ảnh (imutils) chuyển về ảnh xám phục vụ phát hiện
    # khuôn mặt detectMultiScale
    while True:
        frame = vs.read()
        # resize ảnh để tăng tốc độ xử lí
        frame = imutils.resize(frame, width=700)
        # chuyển về ảnh xám phục vụ detectMultiScale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Phát hiện các khuôn mặt có trong hình nhờ detectMultiscale
        faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        # bước 3: duyệt qua lần lượt từng mặt
        # vẽ rectangle và nhận diện các điểm landmask
        for (x, y, w, h) in faces:
            # tạo 1 hcn quanh mặt
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            # nhận diện các điểm landmask
            landmask = landmask_detect(gray, rect)
            # chuyển về numpy
            landmask = face_utils.shape_to_np(landmask)
            # capture vùng miệng
            (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
            mouth = landmask[mStart:mEnd]
            # Lấy hcn bao quanh vùng miêngj
            boundRect = cv2.boundingRect(mouth)
            # các bước từ detectMulti, nhận diện các điểm landmask
            # là làm việc với ảnh xám
            # bây giờ ta sẽ làm việc với ảnh màu và chuyển về HSV
            cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])),
                          (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 0, 255), 2)
            # bước 4: tính saturation trung bình và so sánh
            hsv = cv2.cvtColor(frame[int(boundRect[1]):int(boundRect[1] + boundRect[3]),
                               int(boundRect[0]):int(boundRect[0] + boundRect[2])], cv2.COLOR_BGR2HSV)
            sum_saturation = np.sum(hsv[:, :, 1])
            # area
            area = int(boundRect[2] * boundRect[3])
            avg_saturation = sum_saturation / area
            # kiểm tra với 1 ngưỡng threshold
            pygame.init()
            if avg_saturation > 100:
                cv2.putText(frame, "Deo khau trang vao, toang bay gio", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                pygame.mixer.music.load('khautrang.mp3')

                pygame.mixer.music.play()

            else:
                cv2.putText(frame, "BAN GIU AN TOAN TOT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                pygame.mixer.music.pause()

        # hiển thị lên màn hình
        cv2.imshow("Camera", frame)
        # bấm esc thoát
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()
    vs.stop()
with st.form('form2'):
    new_title = '<p style="font-family:Tahoma;font-weight: 700; color:#0000ff; font-size: 30px;">Hệ thống phát hiện khẩu trang và nhắc nhở nếu không đeo bằng OpenCV</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    new_title1 = '<p style="font-family:Bookman; color:ff7f00; font-size: 18px;">Dịch COVID 19 đang quá căng thẳng nên hôm nay mình xin giới thiệu cách xây dựng một hệ thống phát hiện đeo khẩu trang và nhắc nhở nếu không đeo bằng OpenCV nhé</p>'
    st.markdown(new_title1, unsafe_allow_html=True)
    new_title2 = '<p style="font-family:Bookman; color:ff7f00; font-size: 18px;">Hệ thống của chúng ta sẽ giám sát quá camera, kiểm tra xem người dùng có đeo khẩu trang hay không để thông báo lên màn hình. Trong thực tế chúng ta có thể kết nối ra hệ thống loa để cảnh báo hoặc thông báo cho lực lượng bảo vệ để yêu cầu đeo khẩu trang trước khi vào tòa nhà.</p>'
    st.markdown(new_title2, unsafe_allow_html=True)
    st.write(':mask:'*8)
    st.image(image='images/mask.jpg', caption='mask')
    st.subheader('Phần 1- Cách làm bài toán')
    st.write('Chúng ta sẽ phát hiện khuôn mặt sau đó sử dụng landmark để detect mouth area :nose:. Sau đó tính toán average saturation và compare với 1 threshold do chúng ta đặt ra để check xem có đeo khẩu trang :mask:   hay không. Cách náy có ưu điểm là không cần data, tốc độ chạy cao hơn cách 1 nhưng đôi khi do điều kiện ánh sáng thay đổi thì có thể không detect chuẩn.')
    st.image(image='images/saturation.jpg', caption='point')
    st.subheader('Phần 2 – Clone mã nguồn và cài đặt các thư viện')
    st.write('Các bạn sẽ tiến hành tạo thư mục OPENCV_VA sau đó clone mã nguồn sau về')
    st.code('https://github.com/Anh15052001/OPENCV_VA')
    st.write('Sau đó sẽ tiến hành cài các thư viện cần thiết :thumbsup:')
    st.code('pip install -r setup.txt')
    st.info('Sau khi cài đặt xong rồi chúng ta tiến hành test thử nhé')
    st.subheader('Phần 3 – Chạy thử chương trình phát hiện đeo khẩu trang')
    st.write('Các bạn mở file detect_mask.py trong đó mình đã comment :heart: đầy đủ, mô hình xây dựng của chúng ta như sau: ')
    st.image(image='images/model_mask.png', caption='model')
    st.warning('Tiến hành check thử khi bấm nút Check Mask ngay tại đây :thumbsup:')
    if st.form_submit_button('Check Mask'):
        OpenCV()