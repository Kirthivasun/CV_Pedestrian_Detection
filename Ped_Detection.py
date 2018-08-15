# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

import cv2
import requests

# Replace the URL with your own IPwebcam shot.jpg IP:port
url="/shot.jpg"
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
#args = vars(ap.parse_args())
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
#for imagePath in paths.list_images(args["images"]):

#cap = cv2.VideoCapture(0)

while (True):
	img_resp = requests.get(url)
	img_np = np.array(bytearray(img_resp.content), dtype = np.uint8)
	image = cv2.imdecode(img_np, -1)


	#cv2.imshow('Camera', image)

    #To give the processor some less stress
    #time.sleep(0.1) 

	#ret, frame = cap.read()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
	#if ret==True:	
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	#image = frame.copy()
	#image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(300, image.shape[1]))
	orig = image.copy()
	 
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.01)
	 
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
	# show some information on the number of bounding boxes
	#filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO]: {} original boxes, {} after suppression".format(
		len(rects), len(pick)))
	 
	# show the output images
	#cv2.imshow("Before NMS", orig)
	cv2.imshow("Pedestrian Detection", image)
	#cv2.waitKey(2)
	
#cap.VideoRealease()
cv2.destroyAllWindows()
