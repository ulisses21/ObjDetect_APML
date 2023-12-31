
#Start of Inference code
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import onnxruntime
#load model
model = YOLO('person_w.pt')
#results=model.predict('bus.jpg', save=True)#predict image 
#model.predict('People Walking vid.mp4', save=True, imgsz=(1280,720))
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

