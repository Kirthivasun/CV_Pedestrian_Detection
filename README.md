# CV_Pedestrian_Detection
Model to detect the pedestrian from the video feed

# Python
Developed in Python 3.6.5

 - Numpy 1.13.3
 - OpenCv 3.4.2
 - IMutils 0.4.6
 - Requests 2.18.4
 - Argparse 1.1

# Description:
 - Used the IP camera and connected to camera source to get the video feed.
 - Processed them frame by frame
 - Applied HOG descriptor from CV2 and configured with default pedestrian detector model
 - Predicted the frames from feed using the model
 - Used Non-Maximum suppression from imutils to suppress multiple bounding box detection

