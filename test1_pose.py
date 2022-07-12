import cv2
import mediapipe as mp
import urllib.request
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

img_url = "https://static.turbosquid.com/Preview/2015/11/10__11_56_36/anthonystanding23dmetry3dhuman01.jpg5e774d4d-9b9e-456d-9d7b-fc4d741cf940Large.jpg"
file = 'pose.jpg'
urllib.request.urlretrieve(img_url, file)
# img = Image.open(file)
# img.show()

with mp_pose.Pose(static_image_mode=True,
                  model_complexity=2,
                  enable_segmentation=True) as pose:
    image = cv2.imread(file)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

annotated_image = image.copy()
mp_drawing.draw_landmarks(annotated_image,
                          results.pose_landmarks,
                          mp_pose.POSE_CONNECTIONS,
                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

filename = "pose_wireframe.png"
cv2.imwrite(filename, annotated_image)
Image.open(filename).show()
