# pyrefly: ignore [missing-import]
import streamlit as st
import cv2
import os
import torch
import numpy as np
import tempfile
from ultralytics import YOLO
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception as e:
    import traceback
    with open("import_error.log", "w") as f:
        f.write(traceback.format_exc())
    raise e

st.set_page_config(page_title="Coronary Angiogram Analysis", layout="wide")

# --- Custom Styling for Premium Look ---
st.markdown("""
    <style>
    .stat-box {
        font-size: 26px;
        font-weight: 500;
        margin-bottom: 20px;
        font-family: sans-serif;
    }
    .highlight {
        color: #E2B93B; /* Matches the yellow highlight in your mockup */
        font-weight: bold;
    }
    .header-title {
        text-align: center;
        color: #1E88E5;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header-title">Coronary Angiogram AI Interpretation</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_ai_models():
    """Load models once so the UI stays fast."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "runs", "detect", "stenosis_detector", "weights", "best.pt")
    yolo_model = YOLO(model_path)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    flan_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(flan_name)
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_name).to(device)
    
    return yolo_model, tokenizer, flan_model, device

def preprocess_frame(frame, target_size=(512, 512)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    resized_img = cv2.resize(clahe_img, target_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

def process_single_frame(frame, yolo_model):
    preprocessed = preprocess_frame(frame)
    results = yolo_model.predict(source=preprocessed, conf=0.5, verbose=False)
    result = results[0]
    
    annotated = preprocessed.copy()
    boxes = result.boxes
    
    data = {
        "stenosis_yn": "N",
        "coords": "-",
        "lumen_dia": "-",
        "percentage": "-",
        "raw_conf": 0.0
    }
    
    if len(boxes) > 0:
        data["stenosis_yn"] = "Y"
        # Take the box with highest confidence
        best_box = max(boxes, key=lambda x: float(x.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        conf = float(best_box.conf[0])
        
        data["coords"] = f"[{x1}, {y1}]"
        data["lumen_dia"] = f"{abs(x2 - x1)} pixels"
        data["percentage"] = f"{int(conf * 100)}%"
        data["raw_conf"] = conf
        
        # Draw bounding box (matching the yellow highlight from your mockup)
        color = (59, 185, 226) # BGR for the golden yellow
        thickness = 4
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        label = f"Stenosis: {data['percentage']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - 25), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    return annotated, data

def generate_report(tokenizer, flan_model, device, file_name, findings):
    prompt = (
        "Generate a formal, highly structured clinical radiology report based strictly on the provided data.\n"
        "Do not invent patient names, dates, or specific arteries. Use a professional, objective medical tone.\n\n"
        "DATA:\n"
        f"Procedure: Coronary Angiography\n"
        f"File: {file_name}\n"
        f"AI Detection Results: {findings}\n\n"
        "REQUIRED FORMAT:\n"
        "EXAM: Coronary Angiography\n"
        "FINDINGS: [Clinical description of the AI results]\n"
        "IMPRESSION: [Brief diagnostic summary]\n\n"
        "REPORT:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = flan_model.generate(**inputs, max_new_tokens=200, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 1. Load Models automatically
with st.spinner("Waking up AI Models..."):
    yolo_model, tokenizer, flan_model, device = load_ai_models()

# 2. File Uploader
uploaded_file = st.file_uploader("Upload Angiogram Video or Image", type=["mp4", "avi", "mov", "bmp", "jpg", "png"])

if uploaded_file is not None:
    is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))
    
    with st.spinner("Analyzing Angiogram..."):
        if not is_video:
            # Process single image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            annotated_img, data = process_single_frame(frame, yolo_model)
            
            final_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
        else:
            # Process video - find the frame with the worst stenosis to display on the UI
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Setup output video writer
            out_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video_file.name, fourcc, fps, (512, 512))
            
            best_annotated_img = None
            best_data = {
                "stenosis_yn": "N",
                "coords": "-",
                "lumen_dia": "-",
                "percentage": "-",
                "raw_conf": 0.0
            }
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated_img, data = process_single_frame(frame, yolo_model)
                out.write(annotated_img) # Save to compiled video
                
                # We save the frame that had the highest AI confidence to show the doctor
                if data["raw_conf"] > best_data["raw_conf"]:
                    best_data = data
                    best_annotated_img = annotated_img
                    
            cap.release()
            out.release()
            
            if best_annotated_img is None:
                best_annotated_img = annotated_img # Fallback to last frame
                
            final_img = cv2.cvtColor(best_annotated_img, cv2.COLOR_BGR2RGB)
            data = best_data

        # --- Display Layout (Matching the user's requested Mockup) ---
        col1, col2 = st.columns([0.5, 0.5])
        
        with col1:
            st.image(final_img, caption="AI Highlighted View", use_column_width=True)
            
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-box'>Stenosis : <span class='highlight'>{data['stenosis_yn']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-box'>Coordinate : <span class='highlight'>{data['coords']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-box'>lumen dia : <span class='highlight'>{data['lumen_dia']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-box'>Percentage : <span class='highlight'>{data['percentage']}</span></div>", unsafe_allow_html=True)
            
            # Generate Report
            st.markdown("---")
            st.markdown("### 📝 AI Clinical Report")
            
            if data['stenosis_yn'] == 'Y':
                findings_text = f"Stenosis detected with AI confidence of {data['percentage']}. Estimated narrowing coordinates at {data['coords']} with an approximate lumen bounding box width of {data['lumen_dia']}."
            else:
                findings_text = "No stenosis was detected in the provided angiogram."
                
            report_text = generate_report(tokenizer, flan_model, device, uploaded_file.name, findings_text)
            
            st.success(report_text)
            
            # Provide Download Button for Video
            if is_video:
                st.markdown("---")
                with open(out_video_file.name, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download Processed Angiogram Video",
                        data=vf,
                        file_name=f"AI_Analyzed_{uploaded_file.name}",
                        mime="video/mp4"
                    )
