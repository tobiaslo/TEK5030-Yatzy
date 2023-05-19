import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def detect_dice(blobs, threshold):
    if len(blobs) != 0 and len(blobs) != 1:
        dices = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold).fit(blobs)
        dices = dices.labels_
        return dices
    else:
        return []
    
def draw_dices(frame, dices, keypoints):
    corners = np.zeros(2)

    #print(blobs)

    #zipped = zip(dices, blobs)

    #sort_zip = sorted(zipped, key=lambda x: x[0])

    current_lable = 0
    current_dice = []
    
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []

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
        if len(x) != 0:
            for o in x:
                print(o.size)
                frame = cv2.circle(frame, (int(o.pt[0]), int(o.pt[1])), int(o.size), colors[i], -1)
    
    return frame

    #print(sort_zip)

    #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

def blob(frame):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True 
    params.minCircularity = 0.85

    blob_detector = cv2.SimpleBlobDetector_create(params)

    keypoints = blob_detector.detect(frame)

    y = [x.pt for x in keypoints]
    sizes = [x.size for x in keypoints]
    avg = np.mean(sizes)

    threshold = avg * 7.5

    dices = detect_dice(y, threshold)
    print(dices)

    frame = draw_dices(frame, dices, keypoints)

    return frame


def houghLinesP(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    
    # This returns an array of r and theta values
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=5, maxLineGap=10)

    if lines is None:
        return

    for p in lines:
        x1,y1,x2,y2=p[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

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

    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)
    
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
    
        # x0 stores the value rcos(theta)
        x0 = a*r
    
        # y0 stores the value rsin(theta)
        y0 = b*r
    
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
    
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
    
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
    
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
    
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

def lab00():
    device_id = 0
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print(f'Could not open camera {device_id}')
        return
    else:
        print(f'Successfully opend camera {device_id}')

    window_title = 'Ward linkage and Simple Blob Detector'
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)

    while True:
        success, frame = cap.read()

        if not success:
            print('Image capture did not succeed')
            break

        #houghLinesP(frame)

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #edges = cv2.Canny(gray, 50, 100)
        
        frame = blob(frame)
        
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