from ultralytics import YOLO
# load a custom trained model
model = YOLO('best.pt')  
# Export the model
model.export(format='onnx')