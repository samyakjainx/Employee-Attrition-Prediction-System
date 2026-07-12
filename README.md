# 👨‍💼 Employee Attrition Prediction System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## 📌 Project Overview

Employee attrition is one of the biggest challenges faced by organizations. High employee turnover increases recruitment costs, affects productivity, and impacts overall business performance.

This project uses **Machine Learning** to predict whether an employee is **likely to leave the company or stay**, based on various HR-related factors such as age, monthly income, job satisfaction, overtime, work-life balance, years at the company, and more.

The project includes complete data preprocessing, feature engineering, model comparison, performance evaluation, and deployment using **Streamlit Community Cloud**.

---

# 🚀 Live Demo

🔗 **Web Application**

> **Deployed Streamlit link :**

```
https://employee-attrition-prediction-system-samyakjain.streamlit.app/
```

---

# ✨ Features

- 📊 Interactive Streamlit Web Application
- 🤖 Machine Learning Based Prediction
- 📁 Complete Data Preprocessing Pipeline
- 📈 Employee Attrition Probability
- 🎯 Instant Prediction
- 📋 User Friendly Interface
- 💾 Saved ML Model using Joblib
- 🔄 Automatic Feature Scaling
- 🧠 Label Encoding for Categorical Features
- 📉 Probability Score for Prediction
- ☁️ Deployed on Streamlit Community Cloud

---

# 📂 Project Structure

```
Employee-Attrition-Prediction-System
│
├── Dataset/
│
├── Model/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── features.pkl
│
├── Notebook/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Data_Preprocessing.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Building.ipynb
│   └── 06_Streamlit_Testing.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🧾 Dataset Information

The project is trained on the IBM HR Employee Attrition Dataset.

### Total Employees

- **1470**

### Features

- **30+ Employee Attributes**

### Target Variable

- Attrition

| Value | Meaning |
|--------|----------|
| 0 | Employee will Stay |
| 1 | Employee is likely to Leave |

---

# 📊 Features Used

- Age
- Business Travel
- Daily Rate
- Department
- Distance From Home
- Education
- Education Field
- Environment Satisfaction
- Gender
- Hourly Rate
- Job Involvement
- Job Level
- Job Role
- Job Satisfaction
- Marital Status
- Monthly Income
- Monthly Rate
- Number of Companies Worked
- Over Time
- Percent Salary Hike
- Performance Rating
- Relationship Satisfaction
- Stock Option Level
- Total Working Years
- Training Times Last Year
- Work Life Balance
- Years At Company
- Years In Current Role
- Years Since Last Promotion
- Years With Current Manager

---

# ⚙️ Machine Learning Workflow

```
Raw Dataset
      │
      ▼
Data Understanding
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Saving
      │
      ▼
Streamlit Deployment
```

---

# 🤖 Machine Learning Models Compared

- Logistic Regression ✅
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Random Forest

After comparing all models, **Logistic Regression** achieved the best overall performance and was selected for deployment.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-Learn
- Joblib
- Streamlit
- Git
- GitHub

---

# ▶️ Run Locally

Clone the repository

```bash
git clone https://github.com/samyakjainx/Employee-Attrition-Prediction-System.git
```

Move into the project

```bash
cd Employee-Attrition-Prediction-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📸 Application Preview

- Home Page:
<img width="2628" height="1495" alt="Home Page" src="https://github.com/user-attachments/assets/57801897-2714-47ee-9427-71f6b5e91d47" />

- Employee Input Form:
<img width="2628" height="1495" alt="Employee Input Form" src="https://github.com/user-attachments/assets/9b03f1ee-fea9-4262-99c7-888c93a34b0c" />

- Prediction Result:
<img width="2628" height="1495" alt="Prediction Result" src="https://github.com/user-attachments/assets/2d7f7f70-7dce-4127-9f92-81749ffda6eb" />

- Employee Sumary:
<img width="2628" height="1495" alt="Employee Summary" src="https://github.com/user-attachments/assets/18cf22ea-43ce-4669-89da-53bfa44200f0" />

- Dataset Preview:
<img width="2860" height="1630" alt="Dataset Preview" src="https://github.com/user-attachments/assets/087d94dd-8c41-44e5-95a7-f0d76237326d" />

---

# 🎯 Future Improvements

- Deep Learning Models
- Explainable AI (SHAP)
- Employee Risk Dashboard
- HR Analytics Dashboard
- Database Integration
- PDF Report Generation
- Authentication System

---

# 👨‍💻 Developer

**Samyak Jain**

AI & Machine Learning Enthusiast

GitHub:
https://github.com/samyakjainx

LinkedIn:
https://linkedin.com/in/samyakjain-ai

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates further improvements.

---

# 📜 License

This project is licensed under the MIT License.
