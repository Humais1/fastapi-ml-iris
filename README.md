# FastAPI ML Model - Iris 🌸

This project demonstrates how to deploy a machine learning model using **FastAPI**.  
The model predicts the species of an Iris flower (Setosa, Versicolor, Virginica).

---

## 📌 Problem
- Predict iris species from 4 input features:
  - sepal_length
  - sepal_width
  - petal_length
  - petal_width

---

## 🧠 Model
- Dataset: Scikit-learn Iris dataset  
- Algorithm: Logistic Regression  
- Accuracy: ~93%  

---

## ⚡ API Endpoints
- GET / → Health check  
- GET /model-info → Model metadata  
- POST /predict → Predict a single flower  
- POST /predict-batch → Predict multiple flowers  

---

## 📊 Example

