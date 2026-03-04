from ultralytics import YOLO


if __name__ == "__main__":
    # model = YOLO("model/yolov8n-seg.yaml")
    model = YOLO('ultralytics/cfg/models/v8/yolo-EB.yaml')
    model.load("model/yolov8s-seg.pt")  # 加载预训练分割权重

    results = model.train(
        data="cfg/datasets/fire-1.yaml",
        epochs=300,
        patience=100,
        imgsz=640,
        batch=16,
        workers=0,
        device=0
    )

