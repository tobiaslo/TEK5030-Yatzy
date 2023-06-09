import cv2
import numpy as np
from yatzy import Game, ROUNDS
from PIL import Image, ImageFont, ImageDraw
from sklearn.cluster import AgglomerativeClustering

def detect_dice(blobs):
    if len(blobs) != 0 and len(blobs) != 1:
        distance_thresh = cv2.getTrackbarPos("distance_threshold", "params")
        dices = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_thresh).fit(blobs)
        dices = dices.labels_
        return dices
    else:
        return []
    
def draw_dices(frame, dices, keypoints):
    corners = np.zeros(2)

    current_lable = 0
    current_dice = []
    
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []

    list_of_dices = []

    for i, x in enumerate(dices):
        if x == 0:
            class0.append(keypoints[i])
        elif x == 1:
            class1.append(keypoints[i])
        elif x == 2:
            class2.append(keypoints[i])
        elif x == 3:
            class3.append(keypoints[i])
        elif x == 4:
            class4.append(keypoints[i])
    
    classes = [class0, class1, class2, class3, class4]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (200, 0, 255), (0, 100, 255)]

    for i,x in enumerate(classes):
        length = len(x)
        if length != 0:
            list_of_dices.append(length)
            for o in x:
                frame = cv2.circle(frame, (int(o.pt[0]), int(o.pt[1])), int(o.size), colors[i], -1)
    
    return frame, list_of_dices

def blob(cap):
    success, frame = cap.read()
    contour_image = frame.copy()
    
    if not success:
        cap = cv2.VideoCapture(device_id)
        success, frame = cap.read()
        
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True 
    params.minCircularity = 0.85

    blob_detector = cv2.SimpleBlobDetector_create(params)

    keypoints = blob_detector.detect(frame)

    y = [x.pt for x in keypoints]

    dices = detect_dice(y)

    frame, list_of_dices = draw_dices(frame, dices, keypoints)

    return frame, list_of_dices

def get_blobs_dynamic(frame):
    params = cv2.SimpleBlobDetector_Params()

    circularity = cv2.getTrackbarPos("circularity", "params") / 100.0
    params.filterByCircularity = True 
    params.minCircularity = circularity

    area = cv2.getTrackbarPos("blob_area_min", "params")
    maxarea = cv2.getTrackbarPos("blob_area_max", "params")
    
    params.filterByArea = True
    params.minArea = area
    params.maxArea = maxarea
    
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
    
    keypoints = get_blobs_dynamic(original_frame)
    dice_dots = 0
    for keypoint in keypoints:
        if(cv2.pointPolygonTest(contour, (keypoint.pt), measureDist=False)>=0):
            dice_dots+=1
    
    if dice_dots > 0:
        cv2.putText(out_frame, str(dice_dots), (x+20,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
    return dice_dots


def get_contours(processed_img, original_frame):
    out_frame = original_frame.copy()
    contours, hierarchu = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list_of_dices = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > cv2.getTrackbarPos("contour_area", "params"): #500
            score = dice_score(cnt, out_frame, original_frame)
            if score > 0:
                list_of_dices.append(score)
                cv2.drawContours(out_frame, cnt, -1, (0,255,0),4)
    return out_frame, list_of_dices

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

def getContours(frame, threshold_size=1500):

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)

    threshold_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 9)

    conts = cv2.erode(threshold_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=5)
    imgContour = frame

    contours, hierarchy = cv2.findContours(conts, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = np.array([ cv2.contourArea(cnt) for cnt in contours ])
    areas_over_threshold = np.where(threshold_size < areas)[0]
    areas_under_thrshold = np.where(imgContour.shape[0]*imgContour.shape[1]*0.6 > areas)[0]
    selected_areas = np.intersect1d(areas_over_threshold, areas_under_thrshold)
    num_blobs = []
    centers = []
    dices = []

    keypoints = get_blobs(frame)
    
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
            dices.append(num_blobs)
    return imgContour, dices

def detect_dices_with_morph(cap, show_all=False, show_first_last=False):
    success, frame = cap.read()
    contour_image = frame.copy()
    
    if not success:
        cap = cv2.VideoCapture(device_id)
        success, frame = cap.read()
        
    #Convert original image to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Blur the image to remove noise      
    kernel = cv2.getTrackbarPos("kernel", "params")
    if(kernel%2 == 0):
        kernel+=1
    blur = cv2.medianBlur(gray, kernel)   

    #Find all the dice blobs
    blobs_keypoints = get_blobs_dynamic(blur)
            
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
    remaining_blobs_eroded = cv2.erode(remaining_blobs_dilated, cv2.getStructuringElement(cv2.MORPH_ERODE, (4,4)), iterations=4) #4
    remaining_blobs_opening = cv2.morphologyEx(remaining_blobs_eroded, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5))) #5,5
    
    #Find and fill the new convexhull made by dice "two"
    remaining_convex_blobs = fill_convexhull(remaining_blobs_opening, frame.shape)
    
    #Erode the convexhulls, doing so will enable us to seperate all dices (i.e "sixe's" from "two's")
    convex_dice_dots_eroded = cv2.erode(convex_dice_dots,  cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5)), iterations=2)
    
    #Combine the two blob images and create a binary threshold image
    combined = convex_dice_dots + remaining_convex_blobs
    combined_binary = get_binary_image(combined)
    
    #Now we have defined a perimiter for each dice and we can calculate the score
    contour_image, list_of_dices = get_contours(combined_binary, frame)

    if show_all:
        # #Visualizing the process, from first blobs to score
        out1 = draw_blobs(frame, blobs_keypoints, color=(0, 0, 255))
        out2 = overlay(frame, filled_blobs)
        out3 = overlay(frame, convex_dice_dots)
        out4 = overlay(frame, remaining_blobs_filled_on_black)
        out5 = overlay(frame, remaining_blobs_dilated)
        out6 = overlay(frame, remaining_blobs_opening)
        out7 = overlay(frame, remaining_convex_blobs)
        out8 = overlay(frame, combined)
        out9 = contour_image
        
        #Stack the image, 3x3
        stack1 = np.hstack((out1, out2, out3))
        stack2 = np.hstack((out4, out5, out6))
        stack3 = np.hstack((out7, out8, out9))
        stack = np.vstack((stack1, stack2, stack3))
        window_title = "all"
        cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(window_title,stack)
    
    if show_first_last:
        #First and last
        out1 = draw_blobs(frame, blobs_keypoints, color=(0, 0, 255))
        out9 = contour_image

        cv2.imshow("first_last", np.hstack((out1, out9)))
        
    return contour_image, list_of_dices        


def yatzi():
    #Get camera
    device_id = 0
    cap = get_cap(device_id)

    #Create params window
    cv2.namedWindow("params")
    cv2.resizeWindow("params", 200, 200)
    cv2.createTrackbar("blob_area_min", "params", 115, 10000, PASS)
    cv2.createTrackbar("blob_area_max", "params", 1000, 10000, PASS)
    cv2.createTrackbar("contour_area", "params", 500, 5000, PASS)
    cv2.createTrackbar("circularity", "params", 85, 100, PASS)
    cv2.createTrackbar("convexity", "params", 80, 100, PASS)
    cv2.createTrackbar("kernel", "params", 5, 50, PASS)
    cv2.createTrackbar("distance_threshold", "params", 90, 300, PASS)
    cv2.createTrackbar("version", "params", 0, 1, PASS)

    #Create image processed windwos
    window_title = 'dices'
    window_title2 = 'game'
    
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow(window_title2, cv2.WINDOW_GUI_NORMAL)
    
    current_round = 0
    quit_ = False
    game = Game()
    while True:
        throws = 0
        less_than_five = False
        change = True
        prev_dices = []
        equal = 0
        out_frame = None 
        game_frame = None
        while(throws < 3):
            version = cv2.getTrackbarPos("version", "params")

            if version == 0:
                out_frame, list_of_dices = detect_dices_with_morph(cap, show_all=True)
                limit = 5
                version_string = "Morphology and edge detection"
            else:
                out_frame, list_of_dices = blob(cap)
                limit = 10
                version_string = "Clustering"
            game_frame = Image.fromarray(np.zeros((720,400)))
            
            if len(list_of_dices)<5: 
                less_than_five = True
            
            if(sorted(prev_dices) == sorted(list_of_dices)) and less_than_five:
                equal+=1
            else:
                equal = 0
                change = True
            
            if equal > limit:
                change = False
                equal = 0
                
            if less_than_five and not change and len(list_of_dices)==5:
                throws += 1
                less_than_five = False
                change = True
            
            if(game.round<15):
                interaction = f"{game.get_current_player()}. Round: {ROUNDS[game.round]}, Throw: {throws}"
            else:
                interaction = f"GAME OVER"
            version_string += f" (equal: {equal})"
            cv2.putText(out_frame, str(version_string), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(out_frame, str(interaction), (10,50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
            
            draw = ImageDraw.Draw(game_frame)
            draw.text((10,10), game.__str__())
            game_frame = np.array(game_frame)
            
            cv2.imshow(window_title, out_frame)
            cv2.imshow(window_title2, game_frame)
                        
            prev_dices = list_of_dices
            
            key = cv2.waitKey(15)
            if key == ord('q'):
                quit_ = True
                break                
            
            if key == ord('r'):
                game = Game()
                throws = 0
                less_than_five = False
                change = True
                prev_dices = []
                equal = 0
                        
        if(not game.done):
            game.dice_roll(prev_dices)       
        
        key = cv2.waitKey(15)
        if key == ord('q') or quit_:
            print("Quitting")
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    yatzi()