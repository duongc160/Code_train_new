from ultralytics import YOLO

# =============================
# CONFIG
# =============================
DATA_YAML = r"D:/U2U_Dataset/dataset_80_10_10/dataset.yaml"
MODEL = "yolov8n.pt"     # YOLOv8 nano
IMG_SIZE = 640
EPOCHS = 150
BATCH = 16

# =============================
# TRAIN
# =============================
if __name__ == "__main__":
    model = YOLO(MODEL)  # load pretrained YOLOv8n

    results = model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        name="uav_yolov8n",
        workers=4,
        device=0,        # dùng GPU 0, nếu muốn dùng CPU => "cpu"
        optimizer="AdamW",
        lr0=0.001,       # learning rate
        patience=50,
        cos_lr=True,
        mosaic=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.4,
        shear=0.2,
        flipud=0.0,
        fliplr=0.5,
        mixup=0.0,
    )

    print("🎯 TRAINING DONE!")
