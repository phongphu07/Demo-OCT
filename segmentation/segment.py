from ultralytics import YOLO
import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw
from collections import Counter
def predict(img, st):
    
    model_path = "weights/250430_segment_200epoch_yolov8n_best.pt"
    H, W, _ = img.shape
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model weights from `{model_path}`. The file might be corrupted or missing.\n\n**Error:** {e}")
        return
    results = model.predict(img)
    
    if results[0].masks is None:
        print("\n[INFO]number of masks detected: 0")
        mask_out = np.zeros((H, W), dtype=np.uint8)
        annotated_img = results[0].plot(labels=False, boxes=False)
    else:
        print("\n[INFO]number of masks detected:", len(results[0].masks) )
        
        mask_out =None
        for result in results:
            for _, mask_ in enumerate(result.masks.data):

                mask_gray = mask_.numpy() * 255
                mask_gray = cv.resize(mask_gray, (W, H))
                
                if mask_out is None:
                    mask_out = mask_gray
                else:
                    mask_out = cv.bitwise_or(mask_out, mask_gray)
                
        annotated_img = img.copy()
        check_lumen = 0
        class_counts = Counter()
        for idx, prediction in enumerate(results[0].boxes.xywhn):
            class_id_int = int(results[0].boxes.cls[idx].item())
            class_counts[class_id_int] += 1
            poly = results[0].masks.xyn[idx].tolist()
            poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)
            poly *= [W, H]

            if class_id_int == 0 and check_lumen == 0:
                cv.polylines(annotated_img, [poly.astype('int')], True, (255, 0, 0), 2)
                check_lumen += 1
            elif class_id_int == 1:
                cv.polylines(annotated_img, [poly.astype('int')], True, (0, 255, 0), 2)
            
    st.subheader('Segmentation Summary')
    class_names = model.names
    
    if results[0].masks is None:
        for class_id, class_name in class_names.items():
            st.info(f"✓ No {class_name} detected")
    else:
        for class_id, class_name in class_names.items():
            count = class_counts.get(class_id, 0)
            if count > 0:
                st.success(f"✓ Detected: {count} {class_name}")
            else:
                st.info(f"✓ No {class_name} detected")
                
    st.subheader('Output Image')
    cols = st.columns(2)
    cols[0].image(mask_out, clamp=True, channels='GRAY', use_container_width=True)
    cols[1].image(annotated_img, channels='BGR', use_container_width=True)
    
