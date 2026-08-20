# UIforDeployedSpectrogram
My first deployed spectrogram trained from scratch

# Audio Spectrogram Classification Model

An end-to-end Machine Learning project that demonstrates preprocessing audio signals, converting them into visual spectrograms, training a custom deep learning classification model and deploying the inference engine to production.

## 📌 Project Overview
This repository contains the production code and pipeline setup for deploying an audio classification model trained from scratch. The model processes raw audio input, transforms it into spectrogram representations, and predicts class categories via a lightweight inference API.

* **Model Weights Repository:** [Hugging Face Hub](https://huggingface.co/FisayoF/DeployedSpectrogram)
* **Hosted Demo:** [Render Deployment Link](https://your-app-name.onrender.com)

---

## 🛠️ Tech Stack & Dependencies
* **Frameworks:** PyTorch / TensorFlow, Gradio
* **Audio Processing:** Librosa, Torchaudio
* **Model Hosting:** Hugging Face Hub (`FisayoF/DeployedSpectrogram`)
* **Deployment & Web Hosting:** Render, GitHub

---

## 📂 Repository Structure
```text
├── app.py              # Main application entry point & Gradio UI
├── requirements.txt    # Production dependencies
└── README.md           # Project documentation
