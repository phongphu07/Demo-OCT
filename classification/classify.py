from ultralytics import YOLO
import os
import numpy as np
import cv2 as cv


def to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Chuẩn hóa ảnh từ nhiều định dạng sang BGR 8-bit cho YOLO."""
    if img is None:
        return None
    
    # Chuyển từ 16-bit sang 8-bit nếu cần
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # Chuyển đổi kênh màu
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        if img.shape[2] == 4: # BGRA
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        elif img.shape[2] == 1: # Single channel 3D
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    return img

def predict(img, st):
    model_path = "weights/260324_classification_100epoch_yolo8n.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model weights from `{model_path}`. The file might be corrupted or missing.\n\n**Error:** {e}")
        return
    
    frame = to_bgr_uint8(img)
    if frame is None:
        st.error("Invalid image format.")
        return

    results = model.predict(source=frame, imgsz=224, conf=0.25, verbose=False)[0]
    
    if results.probs is not None:
        pred_idx = results.probs.top1
        pred_label = results.names[pred_idx].upper()
        pred_conf = float(results.probs.top1conf)
        
        annotated_img = results.plot()
        
        st.subheader('Classification Summary')
        st.success(f"✓ Classified as: {pred_label} (Confidence: {pred_conf:.2f})")
        
        st.subheader('Output Image')
        st.image(annotated_img, channels="BGR", use_container_width=True)
    else:
        st.warning("No classification results found.")

