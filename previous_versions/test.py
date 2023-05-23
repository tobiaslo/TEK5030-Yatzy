import cv2
import numpy as np
import math

def blob(frame):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True 
    params.minCircularity = 0.85

    blob_detector = cv2.SimpleBlobDetector_create(params)

    keypoints = blob_detector.detect(frame)

    print(len(keypoints))

    blobs = np.zeros((1,1))
    blobs = cv2.drawKeypoints(frame, keypoints, blobs, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return blobs

# https://discuss.codechef.com/t/how-to-find-angle-between-two-lines/14516/3
def angle(p11, p12, p21, p22):
    m1 = (p12[1] - p11[1]) / (p12[0] - p11[0])
    m2 = (p22[1] - p21[1]) / (p22[0] - p21[0])
    a = math.atan(m1)
    b = math.atan(m2)

    return abs(a - b)

# https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
def intersection(p11, p12, p21, p22):
    a1 = p12[1] - p11[1]
    b1 = p11[0] - p12[0]
    c1 = a1*p11[0] + b1*p11[1]

    a2 = p22[1] - p21[1]
    b2 = p21[0] - p22[0]
    c2 = a2*p21[0] + b2*p21[1]

    det = a1*b2 - a2*b1

    if det == 0:
        return (10**9, 10**9)
    else:
        x = (b2*c1 - b1*c2)/det
        y = (a1*c2 - a2*c1)/det

        return (int(x), int(y))

def houghLinesP(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    
    # This returns an array of r and theta values
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=20, maxLineGap=30)

    if lines is None:
        return
    
    num = 0
    for i in range(len(lines)):
        x1,y1,x2,y2=lines[i][0]

        for j in range(i, len(lines)):
            a = angle((x1, y1), (x2, y2), (lines[j][0][0], lines[j][0][1]), (lines[j][0][2], lines[j][0][3]))

            if a < (math.pi / 2) + 0.01 and a > (math.pi / 2) - 0.01:
                num += 1
                p = intersection((x1, y1), (x2, y2), (lines[j][0][0], lines[j][0][1]), (lines[j][0][2], lines[j][0][3]))
                cv2.circle(frame, p, 3, (0,0, 255), 2)
        
        # Draw the lines joing the points
        # On the original image
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),4)
    print(num)

def houghLines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
    
    # The below for loop runs till r and theta values
    # are in the range of the 2d array

    if lines is None:
        return
    
    used_lines = []
    removed_lines = 0

    for r_theta in lines:
        use = True
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr

        for l in used_lines:
            dr = l[0] - r
            dt = l[1] - theta

            if dr < 0.5 and dt < 0.3:
                use = False
        
        if use == True:
            used_lines.append(arr)
        else:
            removed_lines += 1
            
        if use == True:
            # Stores the value of cos(theta) in a
            a = np.cos(theta)
        
            # Stores the value of sin(theta) in b
            b = np.sin(theta)
        
            # x0 stores the value rcos(theta)
            x0 = a*r
        
            # y0 stores the value rsin(theta)
            y0 = b*r
        
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 2000*(-b))
        
            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 2000*(a))
        
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 2000*(-b))
        
            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 2000*(a))
        
            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #print(f'Removed: {removed_lines}, Drawing: {len(used_lines)}')
    
def squares(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,50,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.8 and ratio <= 1.2:
                frame = cv2.drawContours(frame, [cnt], -1, (0,255,255), 3)
                cv2.putText(frame, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frame = cv2.drawContours(frame, [cnt], -1, (0,255,0), 3)



def lab00():
    device_id = 1
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f'Could not open camera {device_id}')
        return
    else:
        print(f'Successfully opend camera {device_id}')

    window_title = 'Lab 0'
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)

    while True:
        success, frame = cap.read()

        if not success:
            print('Image capture did not succeed')
            break

        houghLinesP(frame)
        #frame = blob(frame) 
        
        cv2.imshow(window_title, frame)
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



