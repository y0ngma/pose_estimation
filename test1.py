from turtle import color
import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import PyQt5
from PIL import Image
# from IPython.display import Video, display
from IPython.display import display
import nb_helpers

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

face_url = "https://1vw4gb3u6ymm1ev2sp2nlcxf-wpengine.netdna-ssl.com/wp-content/uploads/shutterstock_149962697-946x658.jpg"
urllib.request.urlretrieve(face_url, "face_image.jpg")

img = Image.open('face_image.jpg')
img.show()

file = 'face_image.jpg'
# 모든 landmarks에 작은 녹색점 표시할것 설정
drawing_spec = mp_drawing.DrawingSpec(color=[0,255,0], thickness=2, circle_radius=1)

# create a face mesh object
with mp_face_mesh.FaceMesh(
    static_image_mode=True, # True:사진, False:동영상
    max_num_faces=1, # 동시 판별 최대 얼굴
    refine_landmarks=True, # attention mesh모델적용 : landmarks 눈입주변 정제 및 눈동자 추가
    min_detection_confidence=0.5 # 얼굴찾기 threshold
    ) as face_mesh:
    # Read image file with cv2 and process with face_mesh
    image = cv2.imread(file)
    # cv2.imshow('image', image)
    # data gather/preparation, model definition/tuning/testing 없이 바로 사용
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# define boolean corresponding to whether or not a face was detected in the image
face_found = bool(results.multi_face_landmarks)

if face_found:
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        # 랜드마크 얼굴에 그리기
        mp_drawing.draw_landmarks(
            image = annotated_image, # tesselation 입힐 이미지
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION, # 는 점들을 잇는 선정보 frozen set
            landmark_drawing_spec=drawing_spec, # None 하면 landmark에 표시없음
            connection_drawing_spec=mp_drawing_styles 
                .get_default_face_mesh_tesselation_style()) # 점이을때 기본 스타일 사용
        
        # # Draw the facial contours of the face onto the image
        # mp_drawing.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_contours_style())
        
        # # Draw the iris location boxes of the face onto the image
        # mp_drawing.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_iris_connections_style())

    cv2.imwrite('face_tesselation_only.png', annotated_image)
    
img = Image.open('face_tesselation_only.png')
img.show()
# display(img)