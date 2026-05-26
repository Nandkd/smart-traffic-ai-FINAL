# IEEE Abstract — Camera Ready

## AI-Powered Intelligent Traffic Management System Using Machine Learning and Deep Learning

---

**Abstract** — Urban traffic congestion remains a defining challenge of modern smart city infrastructure. This paper presents an AI-Powered Intelligent Traffic Management System (AI-ITMS), a production-grade, full-stack platform that integrates deep learning-based computer vision with ensemble machine learning to deliver real-time, self-adaptive traffic control. The system employs YOLOv8 for five-class vehicle detection—car, motorcycle, bus, truck, and ambulance—achieving a mean Average Precision (mAP@0.5) of 89.1% at 83 frames-per-second on GPU hardware. A custom five-block Convolutional Neural Network (CNN) performs binary ambulance classification with 96.2% accuracy and a ROC-AUC of 0.991, triggering an immediate 90-second priority green phase on the ambulance approach lane. Traffic density is predicted across three congestion classes (Low, Medium, High) using an ensemble of Random Forest (94.2%), XGBoost (95.8%), and Logistic Regression (87.3%) models combined through soft voting, achieving a final accuracy of 96.4% and weighted F1-score of 0.963 on a 15-feature dataset of 15,000 records. Dynamic signal timing is computed per-cycle using ML density outputs and a proportional lane allocation algorithm, reducing simulated average vehicle waiting time by 34.7% and ambulance clearance time by 58.5% relative to fixed-time control. A real-time web application—developed in React 18 with Vite, Tailwind CSS, and Framer Motion on the frontend, and Flask 3.0 with SQLite and JWT authentication on the backend—provides a live monitoring dashboard, predictive analytics with 24×7 congestion heatmaps, per-intersection signal control, and an administrative console. Experimental results demonstrate that AI-ITMS consistently outperforms both static and conventional rule-based adaptive systems across all evaluated metrics. The open architecture supports future integration of reinforcement learning signal controllers, federated cross-city training, and edge-deployed ONNX inference on embedded camera hardware.

---

**Keywords** — Traffic Management, YOLOv8, Convolutional Neural Network, Random Forest, XGBoost, Smart City, Emergency Vehicle Detection, Signal Optimization, Computer Vision, Deep Learning, Flask, React

---

**Index Terms** — I.2.10 Vision and Scene Understanding, I.4.8 Scene Analysis, I.5.4 Applications, J.7 Computers in Other Systems (Transportation)

---

## Authors

**[Author 1]**, Department of Computer Science & Engineering, [Institution Name], [City], [Country]. Email: author1@institution.edu

**[Author 2]**, Department of Computer Science & Engineering, [Institution Name], [City], [Country]. Email: author2@institution.edu

**[Supervisor / Guide]**, Associate Professor, Department of CSE, [Institution], [City]. Email: guide@institution.edu

---

## Paper Metadata

| Field | Value |
|-------|-------|
| Submission type | Final Year Project / Research Paper |
| Conference target | IEEE ICCCNT / ICCV Workshop / IJCAI |
| Manuscript length | 6 pages (IEEE double-column) |
| Figures | 8 (architecture, heatmap, confusion matrices, ROC curves, training curves, comparison chart, signal flow, UI screenshots) |
| Tables | 5 (model comparison, per-class metrics, signal timing, dataset stats, feature list) |
| References | 28 |

---

## CRediT Author Statement

- **Conceptualisation**: All authors
- **Methodology**: Author 1, Author 2
- **Software**: Author 1 (ML pipeline, Flask), Author 2 (React frontend)
- **Validation**: All authors
- **Writing — original draft**: Author 1
- **Writing — review & editing**: Supervisor
- **Supervision**: Supervisor

---

## Acknowledgement

The authors thank [Institution] for providing GPU computing resources and the open-source communities behind Ultralytics YOLOv8, PyTorch, Scikit-learn, and React for making this work possible.

---

## Declaration

This work is original and has not been submitted elsewhere. All datasets used are either publicly available (VisDrone2019, COCO) or synthetically generated. No human subject data was collected.

---

*Formatted for IEEE conference submission — adjust author details before submission.*
