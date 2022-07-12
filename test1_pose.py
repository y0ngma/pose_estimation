import cv2
# import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mputils
from mediapipe.python.solutions import drawing_styles as mpstyles
from mediapipe.python.solutions import pose as mppose
import urllib.request
from PIL import Image

img_url = "https://static.turbosquid.com/Preview/2015/11/10__11_56_36/anthonystanding23dmetry3dhuman01.jpg5e774d4d-9b9e-456d-9d7b-fc4d741cf940Large.jpg"
file = 'pose.jpg'
urllib.request.urlretrieve(img_url, file)
# img = Image.open(file)
# img.show()

with mppose.Pose(static_image_mode=True,
                  model_complexity=2,
                  enable_segmentation=True) as pose:
    image = cv2.imread(file)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

annotated_image = image.copy()
mputils.draw_landmarks(annotated_image,
                          results.pose_landmarks,
                          mppose.POSE_CONNECTIONS,
                          landmark_drawing_spec=mpstyles.get_default_pose_landmarks_style())

filename = "pose_wireframe.png"
cv2.imwrite(filename, annotated_image)
Image.open(filename).show()