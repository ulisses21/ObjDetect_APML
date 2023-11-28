from ultralytics import YOLO
# load a custom trained model
model = YOLO('person_detector_320.pt')  
# Export the model
model.export(format='onnx',imgsz=160)