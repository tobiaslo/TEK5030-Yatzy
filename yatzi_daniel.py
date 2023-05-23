import cv2
import numpy as np
from yatzy import Game, ROUNDS
from PIL import Image, ImageFont, ImageDraw


def get_blobs_dynamic(frame):
    params = cv2.SimpleBlobDetector_Params()

    circularity = cv2.getTrackbarPos("circularity", "params") / 100.0
    params.filterByCircularity = True 
    params.minCircularity = circularity

    area = cv2.getTrackbarPos("blob_area", "params")
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
    
    # keypoints = get_blobs(original_frame)
    keypoints = get_blobs_dynamic(original_frame)
    dice_dots = 0
    for keypoint in keypoints:
        if(cv2.pointPolygonTest(contour, (keypoint.pt), measureDist=False)>=0):
            dice_dots+=1
    
    #Display dice values
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

def detect_dices_with_morph(cap, show_all=True, show_first_last=False):
    success, frame = cap.read()
    contour_image = frame.copy()
    
    if not success:
        cap = cv2.VideoCapture(device_id)
        success, frame = cap.read()
    
    # contour_image = frame.copy()
    
    #Convert original image to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Blur the image to remove noise      
    kernel = cv2.getTrackbarPos("kernel", "params")
    if(kernel%2 == 0):
        kernel+=1
    blur = cv2.medianBlur(gray, kernel)
    # blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    # bilate_blur = cv2.bilateralFilter(gray, kernel, 75, 75)
    

    #Find all the dice blobs
    # blobs_keypoints = get_blobs(blur, circularity=0.65, min_area=90, max_area=1000) #110
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
    # remaining_convex_blobs_eroded = cv2.erode(remaining_convex_blobs,  cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5)), iterations=5)
    convex_dice_dots_eroded = cv2.erode(convex_dice_dots,  cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5)), iterations=2)
    
    #Combine the two blob images and create a binary threshold image
    combined = convex_dice_dots + remaining_convex_blobs
    # combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2)), iterations=1)
    combined_binary = get_binary_image(combined)
    
    #Now we have defined a perimiter for each dice and we can calculate the score
    contour_image, list_of_dices = get_contours(combined_binary, frame)
    # cv2.imshow("test", frame)
    if show_all:
        # #Visualizing the process, from first blobs to score
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
        window_title = "all"
        cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)

        cv2.imshow(window_title,stack)
        # stack7 = np.hstack((blur, medianblur))
        # stack8 = np.hstack((bilate_blur, gray))
        
        # stack9 = np.vstack((stack7, stack8))
        # cv2.imshow("blur",stack9)
    
    if show_first_last:
        #First and last
        out1 = draw_blobs(frame, blobs_keypoints, color=(0, 0, 255))
        out9 = contour_image

        cv2.imshow("first_last", np.hstack((out1, out9)))
        
    return contour_image, list_of_dices        

def is_active_throw():
    pass

def yatzi():
    #Get camera
    device_id = 0
    cap = get_cap(device_id)

    #Create params window
    cv2.namedWindow("params")
    cv2.resizeWindow("params", 200, 200)
    cv2.createTrackbar("blob_area", "params", 115, 10000, PASS)
    cv2.createTrackbar("blob_area_max", "params", 1000, 10000, PASS)

    cv2.createTrackbar("contour_area", "params", 500, 5000, PASS)

    cv2.createTrackbar("minThreshold", "params", 125, 255, PASS)
    cv2.createTrackbar("maxThreshold", "params", 255, 255, PASS)
    cv2.createTrackbar("circularity", "params", 75, 100, PASS)
    cv2.createTrackbar("convexity", "params", 80, 100, PASS)
    
    cv2.createTrackbar("kernel", "params", 5, 50, PASS) #erode: kernel=4 or 5, iter = 5 / 4. Opening: kernel=25 or 5
    cv2.createTrackbar("iterations", "params", 1, 10, PASS)
    
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
        while(throws < 3):
            out_frame, list_of_dices = detect_dices_with_morph(cap, show_first_last=False)
            game_frame = Image.fromarray(np.zeros((720,400)))
            if len(list_of_dices)<5: less_than_five = True
            
            if(sorted(prev_dices) == sorted(list_of_dices)) and less_than_five:
                equal+=1
            else:
                equal = 0
                change = True
            
            if equal > 3:
                change = False
                equal = 0
                
            if less_than_five and not change and len(list_of_dices)==5:
                throws += 1
                less_than_five = False
                change = True
            
            interaction = f"{game.get_current_player()}, Round: {ROUNDS[game.round]}, Throw: {throws}"
            cv2.putText(out_frame, str(interaction), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)

            logic = f"{game.get_current_player()}, less than five: {less_than_five}, change: {change}, equal: {equal}"
            cv2.putText(out_frame, str(logic), (10,50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
            

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
        #Calculate score
        game.dice_roll(prev_dices)
        
        key = cv2.waitKey(15)
        if key == ord('q') or quit_:
            print("Quitting")
            break 


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    yatzi()