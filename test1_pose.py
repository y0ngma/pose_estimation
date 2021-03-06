import cv2
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions import drawing_styles
from mediapipe.python.solutions import pose
from mediapipe.python.solutions import holistic
import nb_helpers

img_url = "https://static.turbosquid.com/Preview/2015/11/10__11_56_36/anthonystanding23dmetry3dhuman01.jpg5e774d4d-9b9e-456d-9d7b-fc4d741cf940Large.jpg"
file = 'pose.jpg'
urllib.request.urlretrieve(img_url, file)
# img = Image.open(file)
# img.show()

with pose.Pose(static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True) as mypose:
    image = cv2.imread(file)
    results = mypose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

annotated_image = image.copy()
drawing_utils.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    pose.POSE_CONNECTIONS,
    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
    )

filename = "pose_wireframe.png"
cv2.imwrite(filename, annotated_image)
Image.open(filename).show()

# pose estimation된 객체에 속할 확률임계치를 넘은 픽셀 빼고 나머지를 배경색으로 칠하기
segmented_image = image.copy()
seg_thres = .3 # 낮을땐 덜 칠해지는 부분이 많은것 확인 가능

# segmentation mask를 3 RGB채널로 stack후 어느픽셀을 유지할지 필터 생성
condition = np.stack((results.segmentation_mask,)*3, axis=-1) > seg_thres

# 칠할 검은배경 생성
bg_image = np.zeros(image.shape, dtype=np.uint8)
# bg_image[:] = [255,0,0] # 다른색으로 칠하기 가능

# if condition: segmented_image else bg_image
segmented_image = np.where(condition, segmented_image, bg_image)

filename = "pose_segmentation.png"
cv2.imwrite(filename, segmented_image)
Image.open(filename).show()


poselandmarks_list = [ repr(item).split('.')[1].split(':')[0] for item in pose.PoseLandmark ]
num = 0
for idx in holistic.POSE_CONNECTIONS:
    if num < 5:
        print(poselandmarks_list[idx[0]], '-->', poselandmarks_list[idx[1]])
    else: break
    num += 1