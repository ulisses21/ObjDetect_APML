from ultralytics import YOLO
import cv2

model = YOLO('person_w.pt')
#model.predict('bus.jpg', save=True)#predict image 
model.predict('People Walking vid.mp4', save=True, imgsz=(1280,720))