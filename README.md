
# 🤖 Robotic Hazard Waste Detection

An **intelligent autonomous system** that leverages **computer vision** and **deep learning** to identify and classify hazardous waste materials in **real-time**. This project is designed to enhance **safety protocols** and streamline **waste management** in hazardous environments.

The system is built using **SSD-MobileNetV2 architecture** for efficient object detection and can be integrated with **Raspberry Pi** and camera modules for practical deployment.

---

## 🧠 Project Overview

The goal of this project is to:

- Detect hazardous waste automatically and in real-time
- Classify different types of hazardous materials
- Reduce human exposure to dangerous environments
- Provide a reliable and automated waste management solution

---

## 🛠 Technologies Used

- **SSD-MobileNetV2** — Efficient object detection architecture  
- **TensorFlow** — Deep learning framework  
- **TensorFlow Object Detection API** — For model training and inference  
- **Raspberry Pi** — For edge deployment  
- **Camera Modules** — Real-time video input  

---

## 📂 Project Structure

```

roboticfinalproject
├── datasetrecord/             # Dataset and TFRecord files
├── exported_model/            # Trained models ready for deployment
├── onnxconversion/            # Scripts to convert models to ONNX
├── pretrained/                # Pretrained SSD-MobileNetV2 models
├── source/                    # Source code for inference and scripts
├── training/                  # Training scripts and configuration
├── tensorboard.txt            # TensorBoard instructions
├── .gitignore
└── README.md

````

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Dararith-ux/roboticfinalproject.git
cd roboticfinalproject
````

### 2️⃣ Install dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

## 🚀 How It Works

1. **Prepare Dataset** — Organize images and annotations in `datasetrecord/`.
2. **Train Model** — Use SSD-MobileNetV2 for object detection using the scripts in `training/`.
3. **Monitor Training** — Visualize metrics with TensorBoard.

```bash
tensorboard --logdir=training
```

4. **Export Model** — Save trained models to `exported_model/`.
5. **Optional Conversion** — Convert TensorFlow models to ONNX for other platforms.
6. **Inference** — Run object detection with scripts in `source/`.

---

## 🧪 Example Usage

```bash
python source/detect.py --model exported_model/ssd_model --input samples/
```

*(Modify paths based on your project structure.)*

---

## 🎯 Features

* Real-time hazardous waste detection
* Efficient object detection using SSD-MobileNetV2
* Raspberry Pi integration for autonomous deployment
* Model export and optional ONNX conversion
* TensorBoard monitoring for training

---

## 📬 Contact

**Dararith**
GitHub: [https://github.com/Dararith-ux](https://github.com/Dararith-ux)
Email: *[your email]*

---

## 📜 License

This project is for **educational and research purposes**.


Do you want me to do that?
```


