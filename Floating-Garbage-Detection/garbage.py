
"""
In summary, this code perform a garbage detection for the project of my client "WATER WASTE GARBAGE COLLECTOR",
itopens your computer's webcam, captures live video, 
performs object detection using a YOLOv4 Tiny model, annotates the detected objects,
 and displays the video with bounding boxes and class labels. 
 The application continues running until you press the 'q' key, after which it releases the resources and exits.
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30) 

classes = []

with open("model/obj.names") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
        
print("Object list")
print(classes)

net = cv2.dnn.readNet("model/custom-yolov4-tiny-detector_last.weights","model/custom-yolov4-tiny-detector.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

while True:
    success, frame = cap.read()
    classIds, scores, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.3)
    
    if len(classIds) != 0:
        for i in range(len(classIds)):
            classId = int(classIds[i])
            confidence = scores[i]
            box = boxes[i]
            x, y, w, h = box
            className = classes[classId-1]
            cv2.putText(frame, className.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cv2.putText(frame, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("Class Ids", classId)
            print("Confidences", confidence)
            print("Boxes", box)

            
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
