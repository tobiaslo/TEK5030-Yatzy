import cv2
import numpy as np
import math

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        corners = cv2.cornerHarris(gray, 25, 3, 0.05)
        corners = cv2.dilate(corners, None)
        frame[corners > 0.01 * corners.max()]=[0, 0, 255]
        
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