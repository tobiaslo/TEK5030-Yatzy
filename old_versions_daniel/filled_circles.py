import cv2
import numpy as np
import math

def get_blobs_dynamic(frame):
    params = cv2.SimpleBlobDetector_Params()

    circularity = cv2.getTrackbarPos("circularity", "params") / 100.0
    params.filterByCircularity = True 
    params.minCircularity = circularity

    area = cv2.getTrackbarPos("blob_area", "params")
    params.filterByArea = True
    params.minArea = area
  
    
    convexity = cv2.getTrackbarPos("convexity", "params") / 100.0
    params.filterByConvexity = True
    params.minConvexity = convexity
    
    blob_detector = cv2.SimpleBlobDetector_create(params)
    keypoints = blob_detector.detect(frame)

    return keypoints

def get_blobs(frame, circularity=0.8, convexity=0.8, min_area=100, max_area=3000):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True 
    params.minCircularity = circularity

    params.filterByArea = True 
    params.minArea =  min_area
    params.maxArea = max_area
    
    params.filterByConvexity = True
    params.minConvexity = convexity
    
    blob_detector = cv2.SimpleBlobDetector_create(params)
    keypoints = blob_detector.detect(frame)

    return keypoints

def get_convexhull(processed_img, out_frame):
    contours, hierarchu = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > cv2.getTrackbarPos("contour_area", "params"):
            convexHull = cv2.convexHull(cnt)
            cv2.drawContours(out_frame, cnt, -1, (0,255,0),4)
            


def get_contours(processed_img, out_frame, original_frame):
    contours, hierarchu = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > cv2.getTrackbarPos("contour_area", "params"):
            cv2.drawContours(out_frame, cnt, -1, (0,255,0),4)
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x,y,w,h = cv2.boundingRect(approx)
            
            keypoints = get_blobs(original_frame)
            num_blobs = 0
            for keypoints in keypoints:
                if(cv2.pointPolygonTest(cnt, (keypoints.pt), measureDist=False)>=0):
                    num_blobs+=1
            
            #Display dice values
            cv2.putText(out_frame, str(num_blobs), (x+20,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)

def fill_blobs(blobs_keypoints, frame, divide_size_by=1):
    out_image = np.zeros(frame.shape, dtype=np.uint8)
    for x in range(0, len(blobs_keypoints)):
        out_image=cv2.circle(out_image, (int(blobs_keypoints[x].pt[0]),int(blobs_keypoints[x].pt[1])), 
                                radius=int((blobs_keypoints[x].size)/divide_size_by), color=(150,150,150), thickness=-1)
    return out_image
    
def merge(blobs_on_frame):
    window_title="distance_trans"
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)

    minThreshold = cv2.getTrackbarPos("minThreshold", "params")
    maxThreshold = cv2.getTrackbarPos("maxThreshold", "params")
    
    gray_img = cv2.cvtColor(blobs_on_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, minThreshold , maxThreshold, cv2.THRESH_BINARY)
    
    # #Determine the distance transform.
    # dist = cv2.distanceTransform(thresh, cv2.DIST_L12, cv2.DIST_MASK_PRECISE)
    # dist_output = cv2.normalize(dist, None, 0, 7.0, cv2.NORM_MINMAX)
    
    # # Make the distance transform normal.
    cv2.imshow(window_title, np.hstack((gray_img, thresh)))
    
    return cv2.merge((thresh,thresh,thresh))
 
    
        

def get_cap(device_id):
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f'Could not open camera {device_id}')
        return
    else:
        print(f'Successfully opend camera {device_id}')
        return cap

        
def draw_blobs(draw_on_this_image, blobs_keypoints, color):
    return cv2.drawKeypoints(draw_on_this_image, blobs_keypoints, None, color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         
def PASS(a):
    pass
       
def get_frame():
    if not success:
        cap = cv2.VideoCapture('IMG_3659.MOV')
        success, frame = cap.read()

def lab00():
    #Get camera
    cap = get_cap(0)

    #Create params window
    cv2.namedWindow("params")
    cv2.resizeWindow("params", 200, 200)
    cv2.createTrackbar("blob_area", "params", 1000, 10000, PASS)
    cv2.createTrackbar("contour_area", "params", 500, 5000, PASS)

    cv2.createTrackbar("minThreshold", "params", 125, 255, PASS)
    cv2.createTrackbar("maxThreshold", "params", 255, 255, PASS)
    cv2.createTrackbar("circularity", "params", 60, 100, PASS)
    cv2.createTrackbar("convexity", "params", 80, 100, PASS)
    
    
    #Create image processed windwos
    window_title = 'before'
    window_title2 = 'after'
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow(window_title2, cv2.WINDOW_GUI_NORMAL)
    
    
    while True:
        #Read image frame
        success, frame = cap.read() #frame is (h x w x 3)
        contour_image = frame.copy()
        
        #Convert to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray is (h x w)
        
        #Blur the image        
        blur = cv2.medianBlur(gray, 7) #blur is (h x w)

        #Get blobs
        blobs_keypoints = get_blobs(gray, circularity=0.75, min_area=110)
        
        #draw blobs
        black_img = np.zeros(frame.shape, dtype=np.uint8)
        blobs_on_black = draw_blobs(black_img, blobs_keypoints, color=(0, 0, 255))
        blobs_on_frame = draw_blobs(frame, blobs_keypoints, color=(0, 0, 255))
        filled_blobs = fill_blobs(blobs_keypoints, frame)
        # filled_blobs = cv2.morphologyEx(filled_blobs, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7)))
        # filled_blobs = cv2.dilate(filled_blobs, cv2.getStructuringElement(cv2.MORPH_DILATE, (7,7)), iterations=1)
        # merged = merge(filled_blobs)
        # edges = cv2.Canny(merged, 100, 255)
        
        edges = cv2.Canny(filled_blobs, 100,255)
        dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=1)

        new_keypoints = get_blobs(edges)
        filled_new_blobs = fill_blobs(new_keypoints, frame, divide_size_by=2)
        
        dilated_filled_new_blobs = cv2.dilate(filled_new_blobs, cv2.getStructuringElement(cv2.MORPH_DILATE, (7,7)), iterations=4)
        eroded = cv2.erode(dilated_filled_new_blobs,  cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5)), iterations=4)
                       
        combined = filled_blobs+eroded
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_DILATE, (5,5)))
        closed_gray = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(closed_gray, 40 , 255, cv2.THRESH_BINARY)

        get_contours(thresh, contour_image, frame)
        # get_contours(thresh, contour_image, frame)
        
        
        
        # print(f"frame shape:{frame.shape}")
        # print(f"len(keypoints):{len(blobs_keypoints)}")
        # print(f"blobs_on_black: {blobs_on_black.shape}")
        # print(f"blobs_on_frame: {blobs_on_frame.shape}")
        # print(f"filled_blobs: {filled_blobs.shape}")
        
        #3channel images
        stack1 = np.hstack((blobs_on_frame,filled_new_blobs))
        stack2 = np.hstack((closed, contour_image))
        cv2.imshow(window_title, np.vstack((stack1,stack2)))
        
        #1channeled
        cv2.imshow(window_title2, dilated)
    

        key = cv2.waitKey(15)
        if key == ord('q'):
            print("Quitting")
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab00()