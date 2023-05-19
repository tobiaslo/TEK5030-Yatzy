import cv2
import numpy as np
import math

def getBlobs(frame):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True 
    params.minCircularity = 0.80

    blob_detector = cv2.SimpleBlobDetector_create(params)

    keypoints = blob_detector.detect(frame)

    #blobs = np.zeros((1,1))
    #blobs = cv2.drawKeypoints(frame, keypoints, blobs, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints

def removeCircle(keypoint, harris):
    l = []
    for i in range(-int(keypoint.size) + 1, int(keypoint.size) - 1):
        for j in range(-int(keypoint.size) + 1, int(keypoint.size) - 1):
            harris[int(keypoint.pt[0]) + i, int(keypoint.pt[1]) + j] = 0.0
    return harris
    
def getContours(conts, imgContour,frame):
    contours, hierarchu = cv2.findContours(conts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    i = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > cv2.getTrackbarPos("area", "Parameters"):
            cv2.drawContours(imgContour, cnt, -1, (0,255,0),4)
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x,y,w,h = cv2.boundingRect(approx)
            
            keypoints = getBlobs(frame)
            num_blobs = 0
            for keypoints in keypoints:
                if(cv2.pointPolygonTest(cnt, (keypoints.pt), measureDist=False)>=0):
                    num_blobs+=1
            
            #Display dice values
            cv2.putText(imgContour, "score "+str(num_blobs), (x+w+20,y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
        i+=1

def PASS(a):
    pass

def lab00():
    device_id = 1
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f'Could not open camera {device_id}')
        return
    else:
        print(f'Successfully opend camera {device_id}')

    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 200, 200)
    cv2.createTrackbar("area", "Parameters", 560, 1000, PASS)
    cv2.createTrackbar("min_pix_val", "Parameters", 125, 255, PASS)
    cv2.createTrackbar("max_pix_val", "Parameters", 255, 255, PASS)
    window_title = 'Lab 0'
    stacked_window = "stacked"
    
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)
    while True:
        success, frame = cap.read()
        imgContour = frame.copy()
        if not success:
            cap = cv2.VideoCapture('IMG_3659.MOV')
            success, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.GaussianBlur(gray, (7,7), 0)

        # alpha = 0.7
        # beta = 50
        # #gray = gray*alpha + beta
        # gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        #Extract data
        corners = cv2.cornerHarris(gray, 25, 3, 0.05)
        blobs_keypoints = getBlobs(gray)
        

        #draw harris
        corner_img = np.zeros(frame.shape, dtype=np.uint8)
        corner_img[corners > 0.005 * corners.max()]=[255, 255, 255]
        # corner_img = cv2.erode(corner_img, np.ones((5, 5)), iterations=5)
        # corner_img = cv2.dilate(corner_img, np.ones((6,6)), iterations=20)

        #draw blobs
        blob = np.zeros((1,1))
        
        blob_img = np.zeros(frame.shape, dtype=np.uint8)
        filled_blob_img = blob_img.copy()
        dilated = blob_img.copy()

        blob = cv2.drawKeypoints(blob_img, blobs_keypoints, blob, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        for x in range(0, len(blobs_keypoints)):
            filled_blob_img=cv2.circle(filled_blob_img, (int(blobs_keypoints[x].pt[0]),int(blobs_keypoints[x].pt[1])), radius=int(blobs_keypoints[x].size), color=(150,150,150), thickness=-1)
            dilated=cv2.circle(dilated, (int(blobs_keypoints[x].pt[0]),int(blobs_keypoints[x].pt[1])), radius=int(blobs_keypoints[x].size), color=(0,100,0), thickness=-1)
        
        dilated = cv2.dilate(dilated,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)), iterations=3)
        eroded = cv2.erode(dilated,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=10)
        
        min_pix_val = cv2.getTrackbarPos("min_pix_val", "Parameters")
        max_pix_val = cv2.getTrackbarPos("max_pix_val", "Parameters")
        
        
        filled_blob_img = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(filled_blob_img,min_pix_val , max_pix_val, cv2.THRESH_BINARY)        
        # blob = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
        #Determine the distance transform.
        dist = cv2.distanceTransform(thresh, cv2.DIST_L12, cv2.DIST_MASK_PRECISE)
        
        # Make the distance transform normal.
        dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        
        
        perimeter = cv2.cvtColor(eroded, cv2.COLOR_BGR2GRAY)
        
 
 
        getContours(gray, imgContour,frame)
        
        # stacked1 = np.hstack((filled_blob_img,(eroded)))
        # stacked2 = np.hstack((dilated,(eroded+dilated+filled_blob_img)))
        # vstk = np.vstack((stacked1,stacked2))
        # cv2.imshow(stacked_window,vstk)



        cv2.imshow("perimiter",dist_output)
        cv2.imshow('Blobs', imgContour)
        # cv2.imshow(window_title, gray)
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