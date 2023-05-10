import cv2
import math
import numpy as np
import imutils

# # Threshold by which lines will be rejected wrt the horizontal
# REJECT_DEGREE_TH = 10.0

# def FilterLines(Lines):
#     FinalLines = []
    
#     for Line in Lines:
#         [[x1, y1, x2, y2]] = Line

#         # Calculating equation of the line: y = mx + c
#         if x1 != x2:
#             m = (y2 - y1) / (x2 - x1)
#         else:
#             m = 100000000
#         c = y2 - m*x2
#         # theta will contain values between -90 -> +90. 
#         theta = np.degrees(math.atan(m))

#         # Rejecting lines of slope near to 0 degree or 90 degree and storing others
#         if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
#             l = np.sqrt( (y2 - y1)**2 + (x2 - x1)**2 )    # length of the line
#             FinalLines.append([x1, y1, x2, y2, m, c, l])

    
#     # Removing extra lines 
#     # (we might get many lines, so we are going to take only longest 15 lines 
#     # for further computation because more than this number of lines will only 
#     # contribute towards slowing down of our algo.)
#     if len(FinalLines) > 15:
#         FinalLines = sorted(FinalLines, key=lambda x: x[-1], reverse=True)
#         FinalLines = FinalLines[:15]
    
#     return FinalLines

# def GetLines(Image):
#     # Converting to grayscale
#     GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
#     # Blurring image to reduce noise.
#     BlurGrayImage = cv2.GaussianBlur(GrayImage, (5, 5), 1)
#     # Generating Edge image
#     EdgeImage = cv2.Canny(BlurGrayImage, 40, 255)

#     # Finding Lines in the image
#     Lines = cv2.HoughLinesP(EdgeImage, 1, np.pi / 180, 50, 10, 15)

#     # Check if lines found and exit if not.
#     if Lines is None:
#         print("Not enough lines found in the image for Vanishing Point detection.")
#         exit(0)
    
#     # Filtering Lines wrt angle
#     FilteredLines = FilterLines(Lines)

#     return FilteredLines

# def GetVanishingPoint(Lines):
#     # We will apply RANSAC inspired algorithm for this. We will take combination 
#     # of 2 lines one by one, find their intersection point, and calculate the 
#     # total error(loss) of that point. Error of the point means root of sum of 
#     # squares of distance of that point from each line.
#     VanishingPoint = None
#     MinError = 100000000000

#     for i in range(len(Lines)):
#         for j in range(i+1, len(Lines)):
#             m1, c1 = Lines[i][4], Lines[i][5]
#             m2, c2 = Lines[j][4], Lines[j][5]

#             if m1 != m2:
#                 x0 = (c1 - c2) / (m2 - m1)
#                 y0 = m1 * x0 + c1

#                 err = 0
#                 for k in range(len(Lines)):
#                     m, c = Lines[k][4], Lines[k][5]
#                     m_ = (-1 / m)
#                     c_ = y0 - m_ * x0

#                     x_ = (c - c_) / (m_ - m)
#                     y_ = m_ * x_ + c_

#                     l = np.sqrt((y_ - y0)**2 + (x_ - x0)**2)

#                     err += l**2

#                 err = np.sqrt(err)

#                 if MinError > err:
#                     MinError = err
#                     VanishingPoint = [x0, y0]
                
#     return VanishingPoint

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
    

        # # Getting the lines form the image
        # Lines = GetLines(frames)

        # # Get vanishing point
        # VanishingPoint = GetVanishingPoint(Lines)

        # # Checking if vanishing point found
        # if VanishingPoint is None:
        #     print("Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point.")
        #     continue
        # else:
        #     cv2.circle(frames, (int(VanishingPoint[0]), int(VanishingPoint[1])), 10, (0, 0, 255), -1)
        #     cv2.putText(frames, f"VP: {VanishingPoint[0]}, {VanishingPoint[1]}", (10, 20), font, fontScale, fontColor, fontThickness)

        
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
