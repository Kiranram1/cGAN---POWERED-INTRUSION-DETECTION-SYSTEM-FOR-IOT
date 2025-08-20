# 🚨 cGAN-Powered Intrusion Detection System for IoT  

![IoT Security](https://img.shields.io/badge/IoT-Security-blue?logo=internetarchive)  
![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)  
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow)  
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green?logo=leaflet)  
![License](https://img.shields.io/badge/License-MIT-red)  

---

## 📌 Overview  
The **Internet of Things (IoT)** is revolutionizing industries but also faces severe **cybersecurity challenges**. Traditional Intrusion Detection Systems (IDS) often fail due to **imbalanced datasets** and **resource limitations**.  

This project introduces a **cGAN-powered IDS** that generates synthetic samples for rare attack classes, combined with a **LightGBM classifier** for efficient multi-class intrusion detection. Evaluated on the **RT-IoT2022 dataset**, our system improves detection rates for minority attacks while staying lightweight and practical for IoT environments.  

---

## 🚀 Features  
✅ **cGAN Data Augmentation** – Balances datasets by generating synthetic samples of rare IoT attacks.  
✅ **Lightweight Detection with LightGBM** – Fast and resource-efficient for IoT devices.  
✅ **Multi-Class Classification** – Detects DoS, DDoS, ARP spoofing, Port Scan, Botnet, and more.  
✅ **Performance Boost** – Higher precision and recall for minority attack classes.  
✅ **Scalable** – Suitable for both real-time and offline IoT network security monitoring.  

---

## 🛠️ Tech Stack  

| Category            | Tools/Frameworks |
|---------------------|------------------|
| **Language**        | ![Python](https://img.shields.io/badge/Python-3.10-yellow?logo=python) |
| **Deep Learning**   | ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow) ![Keras](https://img.shields.io/badge/Keras-red?logo=keras) |
| **ML Model**        | ![LightGBM](https://img.shields.io/badge/LightGBM-green?logo=leaflet) |
| **Data Handling**   | Pandas, NumPy |
| **Visualization**   | Matplotlib, Seaborn |
| **Evaluation**      | Scikit-learn |
| **Dataset**         | [RT-IoT2022](https://www.kaggle.com/datasets/taifurqulfiqar/rtiot2022-dataset) |

---

## 📂 Methodology  

1. **Data Acquisition** – RT-IoT2022 dataset with 123k+ IoT network flow records.  
2. **Preprocessing** – Cleaning, filtering, labeling, feature selection.  
3. **Synthetic Data Generation** – Conditional GAN (cGAN) generates minority attack samples.  
4. **Model Training** – LightGBM trained on augmented dataset.  
5. **Evaluation** – Accuracy, Precision, Recall, and F1-Score measured.  

---

## 📊 Results  

- Improved detection of rare intrusions (e.g., ARP Spoofing, Port Scan).  
- Maintains **>95% accuracy** with low computational overhead.  
- Lightweight enough for **IoT edge devices**.  

---

## 🔮 Future Enhancements  

- Real-time deployment on IoT edge devices.  
- Multi-label attack detection.  
- Integration with cloud/edge security frameworks.  
- Testing on additional datasets (TON-IoT, CICIoT2023).  

---

## 👨‍💻 Authors  

- **Chandan Kumar K R**  
- **Gopinath Ramaje**  
- **Kiran H R**  


