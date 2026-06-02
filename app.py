import streamlit as st
import base64
from st_clickable_images import clickable_images
import cv2 as cv
import numpy as np
from PIL import Image
import detection.detect as detect
import classification.classify as classify
import segmentation.segment as segment





def render_thumbnails(image_paths, key, selected_idx=0):
    st.sidebar.markdown("**Sample Images:**")
    images_b64 = []
    for i, path in enumerate(image_paths):
        try:
            img = cv.imread(path)
            if img is None:
                raise FileNotFoundError
                
            h, w = img.shape[:2]
            _, buffer = cv.imencode('.jpg', img)
            encoded_jpg = base64.b64encode(buffer).decode()
            img_b64 = f"data:image/jpeg;base64,{encoded_jpg}"
            
            # Use SVG for crisp, resolution-independent borders
            if i == selected_idx:
                stroke = "red" 
                stroke_width = "2" 
            else:
                stroke = "white"
                stroke_width = "2"
                
            radius = int(min(w, h) * 0.1) # matches ~8px screen radius
            svg = f'''<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
                <image href="{img_b64}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice"/>
                <rect x="0" y="0" width="{w}" height="{h}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" rx="{radius}" vector-effect="non-scaling-stroke"/>
            </svg>'''
            
            encoded_svg = base64.b64encode(svg.encode('utf-8')).decode()
            images_b64.append(f"data:image/svg+xml;base64,{encoded_svg}")
        except Exception:
            images_b64.append("data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=") # 1x1 transparent
            
    with st.sidebar:
        clicked = clickable_images(
            images_b64,
            titles=[f"Sample {i+1}" for i in range(len(image_paths))],
            div_style={"display": "flex", "justify-content": "space-between", "margin-bottom": "15px"},
            img_style={"width": "30%", "cursor": "pointer", "border-radius": "8px", "object-fit": "cover", "aspect-ratio": "1/1"},
            key=key
        )
    return clicked

def main():
    
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:300px;
            margin-left:-300px;
        }
        </style>
    """,
    unsafe_allow_html=True,
    )
    
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Object Detection', 'Object Classification', "Object Segmentation"])
    
    
    if app_mode == 'About App':
        
        st.header("Introduction to YOLOv8")
        
        
        st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)      
        st.markdown("<p>🚀Welcome to the introduction page of our project! In this project, we will be exploring the YOLO (You Only Look Once) algorithm. YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🎯</p>", unsafe_allow_html=True)  
        st.markdown("<p>The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance. 🌟</p>", unsafe_allow_html=True)
        st.markdown("""<p>🔍Some of these modifications are:<br>
                    &#x2022; Introducing a new backbone network, Darknet-53,<br>
                    &#x2022; Introducing a new anchor-free detection head. This means it predicts directly the center of an object instead of the offset from a known anchor box.<br>
                    &#x2022; and a new loss function.<br></p>""", unsafe_allow_html=True)
        
        st.markdown("""<p>🎊One of the key advantages of YOLOv8 is its versatility. It not only supports object detection but also offers out-of-the-box support for classification and segmentation tasks. This makes it a powerful tool for various computer vision applications.<br><br>
                    ✨In this project, we will focus on three major computer vision tasks that YOLOv8 can be used for: <b>classification</b>, <b>detection</b>, and <b>segmentation</b>. We will explore how YOLOv8 can be applied in the field of medical imaging to detect and classify various anomalies and diseases🧪💊.</p>""", unsafe_allow_html=True)
        
        st.markdown("""<p>We hope you find this project informative and inspiring.💡 Let's dive into the world of YOLOv8 and discover how easy it is to use it!🥁🎆</p>""", unsafe_allow_html=True)
    elif app_mode == "Object Detection":
        
        st.header("Object Detection with YOLOv8",)
        
        st.sidebar.markdown("----")
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.2)
        
        img_file_buffer_detect = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=0)
        
        sample_images = ["DEMO_IMAGES/detect.png", "DEMO_IMAGES/detect4.jpg", "DEMO_IMAGES/detect6.jpg"]
        if "selected_sample_detect" not in st.session_state:
            st.session_state.selected_sample_detect = 0
            
        clicked_idx = render_thumbnails(sample_images, key="img_select_detect", selected_idx=st.session_state.selected_sample_detect)
        
        if clicked_idx > -1 and clicked_idx != st.session_state.selected_sample_detect:
            st.session_state.selected_sample_detect = clicked_idx
            st.rerun()
            
        DEMO_IMAGE = sample_images[st.session_state.selected_sample_detect]
        
        if img_file_buffer_detect is not None:
            img = cv.imdecode(np.frombuffer(img_file_buffer_detect.read(), np.uint8), 1)
        else:
            img = cv.imread(DEMO_IMAGE)
            
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        detect.predict(img, confidence, st)
        
    elif app_mode == "Object Classification":
        
        st.header("Classification with YOLOv8")
        
        st.sidebar.markdown("----")
        
        img_file_buffer_classify = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=1)
        
        sample_images = ["DEMO_IMAGES/classify2.jpg", "DEMO_IMAGES/classify1.jpg", "DEMO_IMAGES/classify3.jpg"]
        if "selected_sample_classify" not in st.session_state:
            st.session_state.selected_sample_classify = 0
            
        clicked_idx = render_thumbnails(sample_images, key="img_select_classify", selected_idx=st.session_state.selected_sample_classify)
        
        if clicked_idx > -1 and clicked_idx != st.session_state.selected_sample_classify:
            st.session_state.selected_sample_classify = clicked_idx
            st.rerun()
            
        DEMO_IMAGE = sample_images[st.session_state.selected_sample_classify]
        
        if img_file_buffer_classify is not None:
            img = cv.imdecode(np.frombuffer(img_file_buffer_classify.read(), np.uint8), 1)
        else:
            img = cv.imread(DEMO_IMAGE)
            
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        classify.predict(img, st)
        
    elif app_mode == "Object Segmentation":
        
        
        st.header("Segmentation with YOLOv8")
        
        st.sidebar.markdown("----")
        
        img_file_buffer_segment = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=2)
        
        sample_images = ["DEMO_IMAGES/segment.png", "DEMO_IMAGES/segment1.jpg", "DEMO_IMAGES/segment2.png"]
        if "selected_sample_segment" not in st.session_state:
            st.session_state.selected_sample_segment = 0
            
        clicked_idx = render_thumbnails(sample_images, key="img_select_segment", selected_idx=st.session_state.selected_sample_segment)
        
        if clicked_idx > -1 and clicked_idx != st.session_state.selected_sample_segment:
            st.session_state.selected_sample_segment = clicked_idx
            st.rerun()
            
        DEMO_IMAGE = sample_images[st.session_state.selected_sample_segment]
        
        if img_file_buffer_segment is not None:
            img = cv.imdecode(np.frombuffer(img_file_buffer_segment.read(), np.uint8), 1)
        else:
            img = cv.imread(DEMO_IMAGE)
            
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        segment.predict(img, st)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
        

