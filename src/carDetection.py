import cv2
import numpy as np
import imutils


if __name__ == "__main__":

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.45
    fontColor = (0, 0, 0)
    fontThickness = 1

    # # Dummy camera intrinsic parameters
    # image_width = 640
    # image_height = 480
    fov = 100.0

    # fx = 6.6051081297156020e+02 # focal_length * sensor_width / image_width
    # fy = 6.6051081297156020e+02 # focal_length * sensor_height / image_height

    # cx = 3.1810845757653777e+02 # image_width / 2
    # cy = 2.3995332228230293e+02 # image_height / 2

    # # Camera intrinsic matrix
    # K = np.array([
    #     [fx, 0, cx],
    #     [0, fy, cy],
    #     [0, 0, 1]
    # ])
    # dist_coeffs = np.array([0., 2.2202255011309072e-01, 0., 0., -5.0348071005413975e-01])

    #cap = cv2.VideoCapture(4)
    cap = cv2.VideoCapture('src/videos/cars.mp4')
    video = True



    car_cascade = cv2.CascadeClassifier('src/cars.xml')
    first = True
    
    while True:
        if video:
            ret, frames = cap.read()
        else:
            frames = cap.copy()

        if first:
            #us plate
            pxPerM = frames.shape[0]/0.30
            pxPlateHightAtCamera = pxPerM*0.15
            pxPlateArea = pxPlateHightAtCamera*frames.shape[0]

            distAtCamera = 0.5*frames.shape[0]/np.tan(np.deg2rad(fov/2)) / pxPerM

            print(f"Px/m: {pxPerM}, pxPlateArea: {pxPlateArea}, distAtCamera: {distAtCamera}")

            first = False

        
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    
        # To draw a rectangle in each cars
        for (x,y,w,h) in cars:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)

            carROI = gray[y:y+h,x:x+w]

            carROI = cv2.bilateralFilter(carROI, 13, 15, 15)
            edged = cv2.Canny(carROI, 30, 200) #Perform Edge detection

            contours=cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
            screenCnt = None

            for c in contours:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                cx, cy, cw, ch = cv2.boundingRect(approx)
                cv2.rectangle(carROI, (cx, cy), (cx+cw, cy+ch), (0, 255, 255), 1)
                if (cy < 0.4*h or (cy+ch) > .8*h) or (cx < 0.3*w or (cx+cw) > 0.7*w ):
                    continue
                if cw >= (ch*1.5) and cw <= (ch*2.5) and (cw*ch) <= (0.05*h*w):  #US Style Plate 0.3x0.15m 
                    cv2.rectangle(carROI, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                    screenCnt = approx
                    cv2.rectangle(frames, (cx+x, cy+y), (cx+cw+x, cy+ch+y), (255, 0, 0), 2)

                    dist = pxPlateArea/(cy*ch)*distAtCamera

                    org = (x, y+ch)
                    cv2.putText(frames, f"Dist: {dist}", org, font, fontScale, fontColor, fontThickness)
                    # break
                elif cw >= (ch*4) and cw <= (ch*5) and (cw*ch) <= (0.05*h*w): #EU Style Plate 0.51x0.12m
                    cv2.rectangle(carROI, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                    screenCnt = approx
                    cv2.rectangle(frames, (cx+x, cy+y), (cx+cw+x, cy+ch+y), (255, 0, 255), 2)

            cv2.imshow('Car ROI', carROI)
            cv2.imshow('Edges', edged)


            
        cv2.imshow('video', frames)

        

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('c'):
            cap = cv2.VideoCapture(4)
            video = True
        if key == ord('1'):
            cap = cv2.VideoCapture('src/videos/cars.mp4')
            video = True
        if key == ord('2'):
            cap = cv2.imread('src/frames/img5.png')
            video = False
        if key == ord(' '): #Pressing space will pause the playback, press any key to resume
            cv2.waitKey(0)

    cv2.destroyAllWindows()
