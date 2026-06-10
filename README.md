# 🫀 Automated Coronary Stenosis Detection & Reporting Pipeline

## 📌 Overview
This project implements an end-to-end deep learning and natural language processing (NLP) pipeline designed to automate the detection of coronary artery stenosis from medical imaging. By processing raw coronary angiogram videos, the system extracts critical frames, performs robust computer vision-based stenosis localization, structures the diagnostic metrics, and leverages a Large Language Model (LLM) to generate comprehensive, clinical-grade medical reports.

The pipeline transitions smoothly from raw, unstructured clinical video data to highly structured, actionable medical intelligence, serving as an advanced diagnostic decision-support system for cardiologists.

## 🎯 Problem Statement
Coronary artery stenosis (the narrowing of heart arteries) is a primary indicator of cardiovascular disease. Manual assessment of coronary angiograms via visual inspection is:
1. **Time-Consuming:** Cardiologists must review dynamic videos frame-by-frame.
2. **Subject to Intra/Inter-Observer Variability:** Qualitative assessments of narrowing percentages can vary significantly between clinicians.
3. **Report Generation Overhead:** Manually compiling visual findings into structured, compliant clinical narratives takes valuable time away from patient care.

This system addresses these challenges by offering an automated, objective, and reproducible framework that detects narrowing, estimates stenosis percentages, and instantly drafts detailed clinical documentation.

## 🏗️ System Architecture
The system is modularly designed into distinct vertical layers to ensure scalability, ease of optimization, and clear separation of concerns.

```
+------------+     +--------------------------+     +---------------------+
|   Input    | --> | Frame Extraction Layer   | --> | Preprocessing Layer |
+------------+     +--------------------------+     +---------------------+
                                                               |
+--------------------------+     +-------------------------+   |
| Report Generation Layer  | <-- | Post-Processing Layer   | <--+
+--------------------------+     +-------------------------+
             |
             v
+--------------------------+
|          Output          |
+--------------------------+
```

### 1. Input Layer
* **Source Data:** Coronary angiogram videos.
* **Supported Formats:** `.mp4`, `.avi`.

### 2. Frame Extraction Layer
* **Core Tool:** OpenCV (`cv2`).
* **Mechanism:** Reads raw angiogram video streams, applies frame-sampling strategies, extracts high-quality sequence frames, and saves them safely to storage.
* **Output:** Sequences of raw angiogram frames (`.jpg` / `.png`).

### 3. Preprocessing Layer
* **Techniques Applied:**
    * **Image Resizing:** Uniformly scales frames to fulfill object detection model input dimensions.
    * **Pixel Normalization:** Standardizes pixel intensity distributions across heterogeneous scanner inputs.
    * **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Maximizes contrast in microvascular areas, enhancing the visibility of blood vessel edges against tissue background.
    * **Noise Reduction:** Attenuates sensor noise and artifacts without blurring clinical edges.
* **Output:** Enhanced, high-contrast angiogram frames.

### 4. Stenosis Detection Layer
* **Framework:** PyTorch.
* **Core Model:** YOLOv8 (You Only Look Once v8).
* **Mechanism:** Executes single-shot object detection across frame sequences to locate narrowing regions, draw spatial bounding boxes, calculate statistical confidence levels, and classify/regress the specific stenosis percentage category.
* **Output:** Detected stenosis locations containing spatial bounding boxes matched with precision confidence scores.

### 5. Post-Processing Layer
* **Logic Operations:**
    * Filters out false-positive detections below a configurable confidence threshold.
    * Tracks and counts distinct stenosis regions over time/frames.
    * Aggregates frame-level estimates to accurately determine the overall stenosis percentage.
    * Formats data arrays into a serialized structural paradigm.
* **Output JSON Schema Example:**
    ```json
    {
      "frame_id": 102,
      "stenosis_detected": true,
      "confidence": 0.91,
      "percentage": 0.70
    }
    ```

### 6. Report Generation Layer
* **Core Model:** Flan-T5 (Fine-tuned Large Language Model).
* **Mechanism:** Consumes the structured post-processing JSON metrics, converts quantitative parameters into fluent medical text narratives, synthesizes clinical findings, and formats a stylized draft medical summary.
* **Output:** Synthesized text-based stenosis report.

### 7. Output Layer
* **Deliverables:**
    * Annotated angiogram frames highlighting spatial bounding boxes.
    * Stenosis detection summaries containing maximum narrowing percentages.
    * AI-generated, structured medical documentation.
* **Export Formats:** Accessible as highly portable PDF files, standardized Text files, or integrated clinical report formats.

## 🛠️ Tools & Technologies
* **Deep Learning & NLP:** PyTorch, YOLOv8, Flan-T5 (Hugging Face Transformers)
* **Computer Vision & Image Processing:** OpenCV, NumPy, Scikit-Image (CLAHE implementation)
* **Data Formatting & Exporting:** Python JSON Serialization, ReportLab / PDF Generation Utilities
* **Techniques:** Object Detection, Feature Standardization, Sequence Analysis, Text Synthesis, Medical Data Restructuring

## 👤 Author
**Harith Ikhwan bin Suhaimi** 📧 [harithikhwan050604@gmail.com](mailto:harithikhwan050604@gmail.com)  
