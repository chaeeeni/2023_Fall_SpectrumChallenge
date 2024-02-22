import cv2
import time
import numpy as np

# mp4 파일 경로
mp4_file = './sunset.mp4'

# 버퍼 사이즈 설정 (1KB)
BUFFER_SIZE = int(256)

# VideoCapture 객체 생성
cap = cv2.VideoCapture(mp4_file)

# VideoCapture 객체가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
f_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
f_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
f_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

#print(fps)

while cap.isOpened():
    ret, frame = cap.read()
    #print(frame)
    if ret:
        re_frame = cv2.resize(frame, (round(f_width), round(f_height)))
        cv2.imshow('Sunset_Video', re_frame)
        key = cv2.waitKey(round(1000/fps))

        if key == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()



# 스트리밍 시작 시간
