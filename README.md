# 🌿 Plant Disease Detection — Streamlit App

Web app that diagnoses plant leaf diseases from a photo using deep learning (PyTorch).
Upload a leaf image → get the crop, the condition, a confidence score and the top-5 predictions.

## Features
- Choose between two trained models: **ResNet18 transfer learning** and a **CNN from scratch**
- 38 classes across 14 crops (healthy + diseased)
- Training curves and model comparison tab

## Results (validation set, 17,572 images)
| Model | Val accuracy |
|---|---|
| ResNet18 features + Logistic Regression (baseline) | 94.79% |
| ResNet18 transfer learning (10 epochs) | 94.49% |
| CNN from scratch (10 epochs) | 93.80% |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
`app.py` UI · `model.py` architectures + inference · `models/` trained weights · `data/` class names + training history

## Dataset & credits
[New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle.
Original model training: Deep Learning course project (instructor: Dr. Tafseer Ahmad) by
Ayesha Akbar Sheikh and Nimra Rasool. Frontend/app: Muhammad Suhail.
