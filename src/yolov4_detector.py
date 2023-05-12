import numpy as np
import imutils
import cv2

inputVideoPath = 'src/videos/cars.mp4'
style = 'US'
outputVideoPath = 'src/videos/output.avi'
yoloWeightsPath = 'src/cfg/yolov4.weights'
yoloConfigPath = 'src/cfg/yolov4.cfg'
detectionProbabilityThresh = 0.5
nonMaximaSuppression = 0.3
labelsPath = 'src/cfg/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(yoloConfigPath, yoloWeightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
vs = cv2.VideoCapture(inputVideoPath)
video = True

first = True

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.45
fontColor = (0, 0, 0)
fontThickness = 1
fov = 100.0

(W, H) = (None,None)
# try:
#     prop = cv2.CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
#     total = int(vs.get(prop))
# except:
#     total = -1
#     print('Frames could not be determined')
while True:
    if video:
        (grabbed, raw) = vs.read()
    else:
        raw = vs.copy()
    frame = raw.copy()
    if not grabbed:
        break

    if first:
        if style=='US':
            #us plate
            pxPerM = frame.shape[0]/0.30
            pxPlateWidthAtCamera = frame.shape[0]
            pxPlateHightAtCamera = pxPerM*0.15
            pxPlateArea = pxPlateHightAtCamera*frame.shape[0]

            distAtCamera = 0.5*frame.shape[0]/np.tan(np.deg2rad(fov/2)) / pxPerM

            print(f"Px/m: {pxPerM}, pxPlateArea: {pxPlateArea}, distAtCamera: {distAtCamera}")

            first = False
        if style=='EU':
            #EU plate
            pxPerM = frame.shape[0]/0.52
            pxPlateWidthAtCamera = frame.shape[0]
            pxPlateHightAtCamera = pxPerM*0.11
            pxPlateArea = pxPlateHightAtCamera*frame.shape[0]

            distAtCamera = 0.5*frame.shape[0]/np.tan(np.deg2rad(fov/2)) / pxPerM

            print(f"Px/m: {pxPerM}, pxPlateArea: {pxPlateArea}, distAtCamera: {distAtCamera}")

            first = False
        if style =='CN':
            #CN plate
            pxPerM = frame.shape[0]/0.44
            pxPlateWidthAtCamera = frame.shape[0]
            pxPlateHightAtCamera = pxPerM*0.14
            pxPlateArea = pxPlateHightAtCamera*frame.shape[0]

            distAtCamera = 0.5*frame.shape[0]/np.tan(np.deg2rad(fov/2)) / pxPerM

            print(f"Px/m: {pxPerM}, pxPlateArea: {pxPlateArea}, distAtCamera: {distAtCamera}")

            first = False
    if W is None and H is None:
        (H, W) = frame.shape[:2]
        
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > detectionProbabilityThresh:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, detectionProbabilityThresh, nonMaximaSuppression)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            lable = LABELS[classIDs[i]]
            text = "{}: {:.4f}".format(lable, confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)# check if the video writer is None

            if lable == "car":
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                carROI = gray[y:y+h,x:x+w]
                if carROI.size < 1:
                    break

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
                    if (cy < 0.4*h or (cy+ch) > h) or (cx < 0.3*w or (cx+cw) > 0.7*w ):
                        continue
                    if (style=='US') and cw >= (ch*1.5) and cw <= (ch*2.5) and (cw*ch) <= (0.05*h*w):  #US Style Plate 0.3x0.15m 
                        cv2.rectangle(carROI, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                        screenCnt = approx
                        cv2.rectangle(frame, (cx+x, cy+y), (cx+cw+x, cy+ch+y), (255, 0, 0), 2)
                        plateROI = raw[cy+y:cy+ch+y, cx+x:cx+cw+x]
                        cv2.imshow('Plate', plateROI)

                        dist1 = pxPlateHightAtCamera/(ch)*distAtCamera
                        dist2 = pxPlateWidthAtCamera/cw*distAtCamera

                        org = (x, y+ch)
                        cv2.putText(frame, f"Dist{(dist1+dist2)/2}", org, font, fontScale, fontColor, fontThickness)
                    elif (style == 'EU') and cw >= (ch*4) and cw <= (ch*5) and (cw*ch) <= (0.05*h*w): #EU Style Plate 0.51x0.12m
                        cv2.rectangle(carROI, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                        screenCnt = approx
                        # cv2.rectangle(frames, (cx+x, cy+y), (cx+cw+x, cy+ch+y), (255, 0, 255), 2)
                    if (style=='CN') and cw >= (ch*2.75) and cw <= (ch*3.5) and (cw*ch) <= (0.05*h*w):  #CN Style Plate 0.44x0.14m 
                        cv2.rectangle(carROI, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                        screenCnt = approx
                        cv2.rectangle(frame, (cx+x, cy+y), (cx+cw+x, cy+ch+y), (255, 0, 0), 2)
                        plateROI = raw[cy+y:cy+ch+y, cx+x:cx+cw+x]
                        cv2.imshow('Plate', plateROI)

                        dist1 = pxPlateHightAtCamera/(ch)*distAtCamera
                        dist2 = pxPlateWidthAtCamera/cw*distAtCamera

                        org = (x, y+ch)
                        cv2.putText(frame, f"Dist{(dist1+dist2)/2}", org, font, fontScale, fontColor, fontThickness)
                        #print(f"D1: {dist1}, d2: {dist2}, avg: {(dist1+dist2)/2}")

                cv2.imshow('Car ROI', carROI)
                cv2.imshow('Edges', edged)
                

    cv2.imshow("detected",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        vs = cv2.VideoCapture(4)
        video = True
    if key == ord('1'):
        vs = cv2.VideoCapture('src/videos/cars.mp4')
        style = 'US'
        video = True
        first = True
    if key == ord('2'):
        vs = cv2.imread('src/frames/img5.png')
        style = 'EU'
        video = False
        first = True
    if key == ord('3'):
        vs = cv2.VideoCapture('src/videos/vid3.mp4')
        fov = 15
        style = 'CN'
        video = True
        first = True
    if key == ord(' '): #Pressing space will pause the playback, press any key to resume
        cv2.waitKey(0)

print("[INFO] cleaning up...")

vs.release()
cv2.destroyAllWindows()