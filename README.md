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

**Request**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

👉 The very last line `EOT` (with no spaces before or after) is required to close the file-writing block.  

Would you like me to also give you the `git add/commit/push` commands so you can send this `README.md` to GitHub right after running this?
