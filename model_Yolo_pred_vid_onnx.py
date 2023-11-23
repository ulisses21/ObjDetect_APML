#install packages
import os
try:
   pip_packages = os.popen("pip list").read()
except Exception:
        print("Error: pip is not installed")
if pip_packages.find("ultralytics" and "opencv-python" and "opencv-contrib-python" and 
                      "torch" and "torchvision" and "torchaudio" and "fiftyone" and "onnx" and 
                      "onnxruntime" and "fiftyone-db-ubuntu2204" and "supervision") != -1:
    print("Required libs are alerty installed")
else:
    os.system("pip3 install ultralytics"and  "pip3 install opencv-python" and "pip3 install opencv-contrib-python" 
    and "pip3 install torch" and "pip3 install torchvision" and "pip3 install torchaudio" 
    and "pip3 install fiftyone" and "pip3 install onnx" and "pip install onnxruntime" and "pip3 install fiftyone-db-ubuntu2204"
    and "pip3 install supervision")
    print("Required libs are installed")

#Start of Inference code
from ultralytics import YOLO
import cv2
import supervision  as sv
import numpy as np
import onnxruntime
#load model
model = YOLO('person_detector.onnx')

video='People Walking vid.mp4'
cap = cv2.VideoCapture(video)

#For visualization of count in screen
Zone_area=np.array([
    [0,0],
    [1280,0],
    [1280,736],
    [0,736]

])
zone=sv.PolygonZone(Zone_area,frame_resolution_wh= (640,640)) 
zone_annotation=sv.PolygonZoneAnnotator(zone=zone,color=sv.Color.red())

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, imgsz=(640),agnostic_nms=True,conf=0.4)
        #get the number of Detected People
        detections = sv.Detections.from_ultralytics(results[0])
        print(len(detections))
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        zone.trigger(detections=detections)
        annotated_frame= zone_annotation.annotate(scene=annotated_frame)
        # Display the annotated frame
        cv2.imshow("Inference",annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
