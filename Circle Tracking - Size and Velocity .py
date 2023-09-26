#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import json

#N.B make sure the video and picture are relating to the same video
#This section must be adjusted for different videos, which may have different framerates and pixel to mm ratios
framerate = 30 # frames per second 
ptomm = 70 # pixels to mm (70:1 capillary)
top = 100 #the top and bottom variables restricts the search area to a segement of the frame
bottom = 650 
radius = []
videoin = r"C:\Users\Simmons\OneDrive\Desktop\example_video.mp4" #change this path to direct to the video you want analyzed 
  
# capture first frame
cam = cv2.VideoCapture(videoin) # upload the video
ret,frame = cam.read()
  
# Release all space and windows once done
#print(pictures)
cam.release()
cv2.destroyAllWindows()

###################################################

# Read image.
img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
# applying different thresholding 
# techniques on the input image
thresh1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
  
thresh2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 199, 5)

#############################################
#img = thresh1
img = frame 

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3,3))
  
# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred, 
    cv2.HOUGH_GRADIENT, 1, 10, param1 = 50,
    param2 = 30, minRadius = int(0.1*ptomm), maxRadius = int(1.5*ptomm))

centers = [[0,0]]
cv2.line(img, (759,50), (848,50), (255,0,0), 2) # draw line(picture,start point, end point, color, thickness) 
cv2.line(img, (759,top), (848,top), (0,0,255), 2)
cv2.line(img, (759,bottom), (848,bottom), (0,0,255), 2) 

# Draw circles that are detected.
if detected_circles is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
  
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
  
        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (100, 200, 0), 2)
        radius.append(r)
        centers.append([a,b])
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        
    #r = radius[0] # show the first box to be used 
    #a = centers[0+1][0]
    #b = centers[0+1][1]
    #bbox = ((a-r), (b-r), 2*r, 2*r) # (x1,y1, x2, y2)
    #p1 = (int(bbox[0]), int(bbox[1]))
    #p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #cv2.rectangle(img, p1, p2, (255,0,0), 2, 1) 
        
    cv2.imshow("Detected Circle", img) # this displays the image with the circles drawn on, if they are correct, hit enter and they will be used as the bounding box definitions for the next section
    cv2.waitKey(0)
    
rad = [x / ptomm for x in radius]

#First image has been processed to identify each circle
#######################################################################        
#Now those circles are fed into the tracker 

circlecount = len(radius)
vel = []
for i in range(circlecount):
    r = radius[i]
    a = centers[i+1][0]
    b = centers[i+1][1]
    x = []
    y = []
    framecount = 0 
    
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
    if __name__ == '__main__' :
 
        # Set up tracker.
        # Instead of CSRT, you can also use
 
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[2]
 
        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                tracker = cv2.legacy_TrackerBoosting()
            elif tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            elif tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            elif tracker_type == 'TLD':
                tracker = cv2.legacy_TrackerTLD()
            elif tracker_type == 'MEDIANFLOW':
                tracker = cv2.legacy_TrackerMedianFlow()
            elif tracker_type == 'GOTURN':
                 tracker = cv2.TrackerGOTURN_create()
            elif tracker_type == 'MOSSE':
                tracker = cv2.legacy_TrackerMOSSE()
            elif tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()
 

        # Read video
        video = cv2.VideoCapture(videoin)
        #video = cv2.VideoCapture(0) # for using CAM
 
        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
 
        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print ('Cannot read video file')
            sys.exit()
        
        # Define an initial bounding box
        bbox = ((a-r), (b-r), 2*r, 2*r) # (x1,y1, x2, y2)
 
        # Uncomment the line below to select a different bounding box
        #bbox = cv2.selectROI(frame, False)
 
        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)
 
        while True:
            # Read a new frame
            ok, frame = video.read()
             
            if not ok:
                break
         
            # Start timer
            timer = cv2.getTickCount()
 
            # Update tracker
            ok, bbox = tracker.update(frame)
            if int(bbox[1])>=bottom : 
                break 
            if int(bbox[1])<=top : 
                break         
         
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                x.append(p1[0])
                y.append(p1[1])
                framecount = framecount +1
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
            # Display result
            cv2.imshow("Tracking", frame)
 
            # Exit if ESC pressed
            if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
                break
        video.release()
        cv2.destroyAllWindows()
        if framecount>(framerate*1.5) : #this ensures very short sets of data aren't considered 
            deltax = (x[0] - x[-1])/ptomm #distance in mm
            deltay = (y[-1] - y[0])/ptomm #distance in mm
            deltat = framecount/framerate #time in sec 
            vel.append((np.sqrt(deltax**2 + deltay**2)) / deltat)
print(vel)
print(rad)

