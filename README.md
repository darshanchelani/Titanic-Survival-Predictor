
# 🚢 Titanic Survival Predictor

**Predicting Passenger Survival on the Titanic using Machine Learning Pipelines**

---

## Project Overview

This project uses the famous **Titanic dataset** from [Kaggle](https://www.kaggle.com/c/titanic) to predict which passengers survived. The focus is on building a **professional, reproducible machine learning workflow**:

- Clean and preprocess data
- Engineer predictive features
- Build ML pipelines to prevent data leakage
- Train and evaluate models with clear metrics
- Save the pipeline for future predictions

---

## Dataset

- **Source:** Kaggle Titanic Competition  
- **Files Used:**  
  - `train.csv` → used for training & validation  
  - `test.csv` → used for final predictions  

**Columns Overview:**

| Feature       | Description |
|---------------|-------------|
| PassengerId   | Unique ID |
| Pclass        | Passenger class (1, 2, 3) |
| Name          | Full name |
| Sex           | Male/Female |
| Age           | Age in years |
| SibSp         | Number of siblings/spouses aboard |
| Parch         | Number of parents/children aboard |
| Ticket        | Ticket number |
| Fare          | Passenger fare |
| Cabin         | Cabin number |
| Embarked      | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
| Survived      | Target (0 = No, 1 = Yes) |

---

## Project Steps

### 1️⃣ Data Loading & EDA
- Loaded `train.csv` and `test.csv`
- Visualized relationships:
  - Survival vs Sex
  - Survival vs Pclass
  - Correlation matrix of numerical features

### 2️⃣ Feature Engineering
- Extracted **Title** from `Name` (Mr, Mrs, Miss, etc.)
- Grouped rare titles into a single category `Rare`
- Dropped irrelevant features: `PassengerId`, `Name`, `Ticket`, `Cabin`

### 3️⃣ Preprocessing Pipelines
- **Numerical Features:** `Age`, `SibSp`, `Parch`, `Fare`  
  - Missing values imputed with median  
  - Standard scaling applied  

- **Categorical Features:** `Sex`, `Pclass`, `Embarked`, `Title`  
  - Missing values imputed with mode  
  - One-hot encoding applied  
  - `handle_unknown='ignore'` ensures robust handling of unseen categories  

- **Combined using `ColumnTransformer`**  

### 4️⃣ Model Training
- Random Forest Classifier (`n_estimators=100`, `max_depth=5`)  
- Training/validation split: 80% / 20% stratified  
- Pipeline ensures **no data leakage**  

### 5️⃣ Model Evaluation
- **Validation Accuracy:** 82.7%  
- **Precision / Recall / F1-score:**  
  - Class 0: precision 0.82, recall 0.92  
  - Class 1: precision 0.84, recall 0.68  
- Confusion matrix plotted for interpretation  
- Feature importance visualized  

### 6️⃣ Model Persistence
- Full pipeline saved as `titanic_model_pipeline.pkl`  
- Ready for deployment or future predictions  

```python
import joblib
model = joblib.load("titanic_model_pipeline.pkl")
predictions = model.predict(new_data)
````

---

## Technologies Used

* Python 3
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-Learn (`Pipeline`, `RandomForestClassifier`, `ColumnTransformer`)

---

---

### Author

* Darshan Chelani
* GitHub: [https://github.com/darshanchelani](https://github.com/darshanchelani)

```

