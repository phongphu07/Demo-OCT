from ultralytics import YOLO
import os
import cv2 as cv
from PIL import Image
from collections import Counter
from collections import Counter


def predict(img, confidence, st):
    # detection model
    model_path = "weights/250911_detect_400epoch_yolov8n_best.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model weights from `{model_path}`. The file might be corrupted or missing.\n\n**Error:** {e}")
        return
     
    # Predict
    results = model.predict(img, conf=confidence)
    result = results[0]
    
    print("\n[INFO] Number of objects detected : ", len(result.boxes) )
    
    annotated_img = img.copy()
    dh, dw, _ = annotated_img.shape

    boxes = []
    class_counts = Counter()
    for idx, prediction in enumerate(result.boxes.xywhn):
        cl = int(result.boxes.cls[idx].item())
        x, y, w, h = prediction.tolist()
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        boxes.append((cl, max(0, l), max(0, t), min(dw - 1, r), min(dh - 1, b)))
        class_counts[cl] += 1

    for cl, x1, y1, x2, y2 in boxes:
        color = (0, 255, 0) if cl == 0 else (255, 0, 255)
        thickness = 2 if cl == 0 else 3
        cv.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
        
        # Optional: Add label text
        class_name = model.names[cl]
        label = f"{class_name}"
        cv.putText(annotated_img, label, (x1, max(y1 - 10, 0)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    st.subheader('Detection Summary')
    class_names = model.names 
    for class_id, class_name in class_names.items():
        count = class_counts.get(class_id, 0)
        if count > 0:
            st.success(f"✓ Detected: {count} {class_name}")
        else:
            st.info(f"✓ No {class_name} detected")
            
    st.subheader('Output Image')
    st.image(annotated_img, channels="BGR", use_container_width=True)