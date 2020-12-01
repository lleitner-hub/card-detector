
#source: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

import cv2
import numpy as np
import os
from ipython_genutils.py3compat import xrange
from matplotlib import pyplot as plt

print(cv2.__version__)

# create video object: argument is the device index (usually 0 if only one video device is present. also a video file
# can be provided
cap = cv2.VideoCapture(0)

sift = cv2.xfeatures2d.SIFT_create()

path = 'img'
class_img_list = os.listdir(path)
MIN_MATCH_COUNT = 20

# get all images from folder 'img' and make an model[keypoints, descriptor, name, image] array
def getModels():
    temp_models = []
    for class_img in class_img_list:
        img = cv2.imread(path+'/'+ class_img)
        kp, dsc = sift.detectAndCompute(img, None)
        temp_models.append([kp, dsc, class_img, img])

    return temp_models


img_models = getModels()

for model in img_models:
    descriptor = model[1]
    imgName = model[2]
    print(imgName + ' Descriptor dimension: ')
    print(np.shape(descriptor)) # dimensionality of the feature descriptor

while(True):

    # capture video stream frame-by-frame: cap.read() returns a bool (True/False) and a frame
    ret, frame = cap.read()

    if ret == False:
        print('No data in frame.')
        continue

    keypoints_frame, descriptor_frame = sift.detectAndCompute(frame, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # for each model check if model is in the frame
    for model in img_models:
        keypoints_model = model[0]
        descriptor_model = model[1]
        text_model = model[2]
        img_model = model[3]


        matches = flann.knnMatch(descriptor_model, descriptor_frame, k=2)

        good_matches = []
        # because of knnMatch ( .. , k=2) --> you get m,n of the two descriptors
        # only get the good matches. 0.75 from docu
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)


        if len(good_matches) > MIN_MATCH_COUNT:
            try:
                src_pts = np.float32([keypoints_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                matchesMask = mask.ravel().tolist()
                h, w, d = img_model.shape

                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # draw your rectangle and text on the frame
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                frame = cv2.putText(frame, text_model, (dst[2][0][0],dst[2][0][1]) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            except:
                pass
    # display the frame, quit video stream by pressing 'q'
    cv2.imshow('Find Card', frame)

    # To quit the live stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # To capture a frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.waitKey(0)
        cv2.imwrite('modelimg.png', frame)
        print('Frame captured...')

# before exiting the program, release the capture
cap.release()
cv2.destroyAllWindows()

