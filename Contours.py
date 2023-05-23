import cv2
import numpy as np
import math

def getBlobs(frame):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True 
    params.minCircularity = 0.5

    blob_detector = cv2.SimpleBlobDetector_create(params)

    keypoints = blob_detector.detect(frame)

    blobs = np.zeros((1,1))
    blobs = cv2.drawKeypoints(frame, keypoints, frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints

def getContours(conts, imgContour,frame, threshold_size=1500):
    contours, hierarchy = cv2.findContours(conts, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = np.array([ cv2.contourArea(cnt) for cnt in contours ])
    areas_over_threshold = np.where(threshold_size < areas)[0]
    areas_under_thrshold = np.where(imgContour.shape[0]*imgContour.shape[1]*0.6 > areas)[0]
    selected_areas = np.intersect1d(areas_over_threshold, areas_under_thrshold)
    num_blobs = []
    centers = []

    keypoints = getBlobs(frame)
    
    areas_over_threshold = selected_areas
    for i in areas_over_threshold:
        cnt = contours[i]
        cv2.drawContours(imgContour, cnt, -1, (0,255,0),3)
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        center = (int(x + w/2), int(y + h/2))

        #cv2.circle(imgContour, center, 15, (255, 0, 0), -1)
        
        
        num_blob = 0
        for keypoint in keypoints:
            if(cv2.pointPolygonTest(cnt, (keypoint.pt), measureDist=False)>=0):
                num_blob+=1
        
        #cv2.putText(imgContour, f"score {i} {num_blob}", center, cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
        
        num_blobs.append(num_blob)
        centers.append(center)
            
            #Display dice values
    for a_i, i in enumerate(areas_over_threshold):
        s = True
        for a_j, j in enumerate(areas_over_threshold):
            if i != j and (cv2.pointPolygonTest(contours[i], (centers[a_j]), measureDist=False) >= 0) and num_blobs[a_j] > 0 and areas[i] > areas[j]:
                s = False
        
        if s and num_blobs[a_i] > 0:
            cv2.drawContours(imgContour, contours[i], -1, (0,255,255),6)
            cv2.putText(imgContour, f"score {num_blobs[a_i]}", centers[a_i], cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)

def lab00():
    device_id = 1
    #cap = cv2.VideoCapture('IMG_3659.MOV')
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f'Could not open camera {device_id}')
        return
    else:
        print(f'Successfully opend camera {device_id}')

    while True:
        success, frame = cap.read()

        if not success:
            cap = cv2.VideoCapture('IMG_3659.MOV')
            success, frame = cap.read()

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)
        gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)
        gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)

        threshold_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 9)

        threshold_img = cv2.erode(threshold_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=5)

        blobs_keypoints = getBlobs(gray_img)
        for keypoint in blobs_keypoints:
            threshold_img = cv2.circle(threshold_img, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size), 1.0, -1)

        getContours(threshold_img, frame, frame)
        #contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contour_img = cv2.drawContours(frame, contours, -1, (0, 255, 75), 2)

        cv2.imshow('threshold_img', threshold_img)
        #cv2.imshow('gray', gray_img)
        cv2.imshow('frame', frame)
        #cv2.imshow('Contours', contour_img)

        delay_ms = 15
        key = cv2.waitKey(delay_ms)

        # React to keyboard commands.
        if key == ord('q'):
            print("Quitting")
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab00()