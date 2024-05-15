import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchvision.models as models


import sys
sys.path.append('C:/Users/kalin/코딩/kickboard2/models')

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 모델 경로 설정 및 모델 로드
model_path = 'C:/Users/kalin/코딩/kickboard2/weight.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path ='C:/Users/kalin/코딩/kickboard2/weight.pt')

# 원래대로 PosixPath 복원
pathlib.PosixPath = temp

#from pathlib import Path
#model_path = Path('C:/Users/kalin/코딩/kickboard/kickboard/weight.pt')
#model = torch.load(model_path) #모델 로드
#model.eval()


tracker = DeepSort(max_age = 30, nn_budget=70, override_track_class=None) #DeepSort 추적기 초기화
#max_age는 추적할 객체 검출되지 않을때 추적 유지하는 시간 결정 nn_budget는 추적기의 성능과 정확도를 조절


path = 'kickboard.mp4'

cap = cv2.VideoCapture(path)

while True:   #탐지 결과 순회하며 각 객체의 정보(바운딩 박스, 신뢰도, 클래스)를 리스트에 추가
    ret, frame = cap.read()  #비디오로부터 다음 프레임 읽어오기 
    if not ret: #프레임 읽기 성공 여부
        break

    results = model(frame)  #yolov5 모델 사용 현재 프레임에서 객체 탐지
    detections = []  #탐지된 객체 정보 저장할 리스트

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        detections.append([[x1, y1, x2, y2],conf, cls])

    tracker_outputs = tracker.update_tracks(detections, frame=frame)


    #Deep Sort 사용하여 탐지된 객체 추적
    for track in tracker_outputs:
        bbox = track.to_tlbr()
        id = track.track_id
        cls = model.names[int(track.det_class)]
        # cls = model.names[int(track.get_class())]

        #각 객체의 바운딩 박스와 아이디, 클래스 이름 그리기
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(frame, f" {cls} {id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow('YOLOv5 DeepSORT Tracking', frame) #open cv 이용 추적 결과가 포함된  프레임 디스플레이

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #자원 정리 
cv2.destroyAllWindows()