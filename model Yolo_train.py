from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt') # pre-trained model
train_results = model.train(data="custom_dataset.yaml", epochs=100, imgsz=320, device=[0,1])#train with dataset from OpenImagesV7 only person class
model.val()


