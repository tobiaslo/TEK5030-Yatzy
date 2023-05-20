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

def get_blobs(frame, circularity=0.8, convexity=0.8, min_area=100, max_area=2000):
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

def fill_convexhull(processed_img, shape):
    out_frame = np.zeros(shape, dtype=np.uint8)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchu = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        conv = cv2.convexHull(cnt)
        cv2.drawContours(out_frame, [conv], -1, (0,0,255),thickness=-1)
    return out_frame
            
def dice_score(contour, out_frame, original_frame):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02*peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    
    keypoints = get_blobs(original_frame)
    num_blobs = 0
    for keypoints in keypoints:
        if(cv2.pointPolygonTest(contour, (keypoints.pt), measureDist=False)>=0):
            num_blobs+=1
    
    #Display dice values
    cv2.putText(out_frame, str(num_blobs), (x+20,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)


def get_contours(processed_img, out_frame, original_frame):
    contours, hierarchu = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > cv2.getTrackbarPos("contour_area", "params"):
            cv2.drawContours(out_frame, cnt, -1, (0,255,0),4)
            dice_score(cnt, out_frame, original_frame)

def fill_blobs(blobs_keypoints, shape, divide_size_by=1):
    out_image = np.zeros(shape, dtype=np.uint8)
    for x in range(0, len(blobs_keypoints)):
        out_image=cv2.circle(out_image, (int(blobs_keypoints[x].pt[0]),int(blobs_keypoints[x].pt[1])), 
                                radius=int((blobs_keypoints[x].size)/divide_size_by), color=(150,150,150), thickness=-1)
    return out_image

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
        
def get_binary_image(image, minThres=40, maxThres=255):
    if(len(image.shape)>2):
       image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(image, minThres , maxThres, cv2.THRESH_BINARY)
    return thresh

def overlay(frame, mask):
    image = frame.copy()
    mask = get_binary_image(mask)
    image[mask==255] = (50,255,50)
    return image
    
def yatzi():
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
    
    cv2.createTrackbar("kernel", "params", 25, 50, PASS) #erode: kernel=4 or 5, iter = 5 / 4. Opening: kernel=25 or 5
    cv2.createTrackbar("iterations", "params", 1, 10, PASS)
    
    #Create image processed windwos
    window_title = 'before'
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)

    while True:
        #Read image frame
        success, frame = cap.read()
        contour_image = frame.copy()

        #Convert original image to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Blur the image to remove noise      
        blur = cv2.medianBlur(gray, 7)

        #Find all the dice blobs
        blobs_keypoints = get_blobs(gray, circularity=0.75, min_area=110)
                
        #Fill the blobs, which are currently only framed by their edges, this results in a merge of all dices, except "one's" and "two's"
        filled_blobs = fill_blobs(blobs_keypoints, frame.shape)
        
        #Find and fill the current convexhulls, for each dice (expect the dice "two" and "one")
        convex_dice_dots = fill_convexhull(filled_blobs, frame.shape)
        edges_from_covexhulls = cv2.Canny(convex_dice_dots, 100, 255)
        
        #Find the remaining blobs, which should only be dice "one" and "two"
        remaining_blobs_keypoints = get_blobs(edges_from_covexhulls)
        
        #Fill the remaining blobs
        remaining_blobs_filled_on_black = fill_blobs(remaining_blobs_keypoints, frame.shape, divide_size_by = 2)

        #Increase the remaining circles by dilation, in order for them to touch
        remaining_blobs_dilated = cv2.dilate(remaining_blobs_filled_on_black, cv2.getStructuringElement(cv2.MORPH_DILATE, (3,3)), iterations=10)
        
        #Remove neighboring "two" dices by performing eroding then opening
        remaining_blobs_eroded = cv2.erode(remaining_blobs_dilated, cv2.getStructuringElement(cv2.MORPH_ERODE, (4,4)), iterations=4)
        remaining_blobs_opening = cv2.morphologyEx(remaining_blobs_eroded, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5)))
        
        #Find and fill the new convexhull made by dice "two"
        remaining_convex_blobs = fill_convexhull(remaining_blobs_opening, frame.shape)
        
        #Erode the convexhulls, doing so will enable us to seperate all dices (i.e "sixe's" from "two's")
        # remaining_convex_blobs_eroded = cv2.erode(remaining_convex_blobs,  cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5)), iterations=5)
        convex_dice_dots_eroded = cv2.erode(convex_dice_dots,  cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5)), iterations=2)
        
        #Combine the two blob images and create a binary threshold image
        combined = convex_dice_dots_eroded + remaining_convex_blobs
        combined_binary = get_binary_image(combined)
        
        #Now we have defined a perimiter for each dice and we can calculate the score
        get_contours(combined_binary, contour_image, frame)
        
        #Visualizing the process, from first blobs to score
        out1 = draw_blobs(frame, blobs_keypoints, color=(0, 0, 255))
        out2 = overlay(frame, convex_dice_dots)
        out3 = overlay(frame, remaining_blobs_filled_on_black)
        out4 = overlay(frame, remaining_blobs_dilated)
        out5 = overlay(frame, remaining_blobs_opening)
        out6 = overlay(frame, remaining_convex_blobs)
        out7 = overlay(frame, convex_dice_dots_eroded)
        out8 = overlay(frame, combined)
        out9 = contour_image
        
        #Stack the image, 3x3
        stack1 = np.hstack((out1, out2, out3))
        stack2 = np.hstack((out4, out5, out6))
        stack3 = np.hstack((out7, out8, out9))
        stack = np.vstack((stack1, stack2, stack3))
        
        cv2.imshow(window_title,stack)

        key = cv2.waitKey(15)
        if key == ord('q'):
            print("Quitting")
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    yatzi()