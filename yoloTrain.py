from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='trainRun.yaml', epochs=100, imgsz=640, project='./trainingData', name='Run1')


