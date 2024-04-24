from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False

    model = YOLO("yolov8n.yaml").to(device)
    model.train(data="data.yaml", epochs=100)
    path = model.export(format="onnx")