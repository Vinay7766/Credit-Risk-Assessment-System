# Credit-Risk-Assessment-System
An AI-powered fintech platform that predicts loan default risk and optimizes interest rates using GenAI, Neural Networks, and Streamlit.

# 💳 CreditIQ | AI-Powered Fintech Risk Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![AI](https://img.shields.io/badge/GenAI-Google%20Gemini-green)
![License](https://img.shields.io/badge/License-MIT-purple)

## 📌 Overview
**CreditIQ** is an end-to-end machine learning application designed to modernize the loan underwriting process. It serves two main purposes:
1.  **Risk Assessment:** Predicts the probability of a borrower defaulting using Logistic Regression (80:20 split).
2.  **Dynamic Pricing:** Suggests an optimal interest rate using a Neural Network (MLP) Regressor.

The system features an **AI Assistant** (powered by Google Gemini 2.5 Flash) that explains decisions in plain English and an **Auto-Solver** that suggests how high-risk applicants can get approved.

## 🚀 Key Features
* **Dual-Model Architecture:** Classification (Default/No Default) & Regression (Interest Rate).
* **Generative AI Chatbot:** integrated fintech advisor for real-time Q&A.
* **Explainable AI (XAI):** SHAP values visualize feature impact (why a loan was rejected).
* **Audio Narration:** Text-to-Speech (TTS) engine narrates the risk analysis.
* **Interactive Dashboard:** Built with Streamlit for a responsive, dark-mode-enabled UI.

## 🛠️ Technologies Used
* **Frontend:** Streamlit
* **ML Core:** Scikit-Learn (Logistic Regression, MLP Regressor)
* **GenAI:** Google Gemini API (v2.5 Flash)
* **Explainability:** SHAP (Shapley Additive Explanations)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## ⚙️ Installation & Usage
Follow these steps to set up and run the project locally.

### 1. Prerequisites
* Python 3.8 or higher
* A Google Gemini API Key (Get one [here](https://aistudio.google.com/app/apikey))


### 2. **Clone the repo:**
   ```bash
   git clone [https://github.com/Vinay7766/Credit-Risk-Assessment-System.git](https://github.com/Vinay7766/Credit-Risk-Assessment-System.git)
   cd Credit-Risk-Assessment-System  
```
### 3. Set up a Virtual Environment (Recommended)
It is best practice to use a virtual environment to manage dependencies.

Windows:
Bash 
python -m venv venv
venv\Scripts\activate

Mac/Linux:
Bash
python3 -m venv venv
source venv/bin/activate

### 4. Install Dependencies
Bash
pip install -r requirements.txt

### 5. Generate Model Artifacts

Note: This project requires pre-trained models and SHAP data to run. You must generate them first.

Open the file Training_Notebook.ipynb (or your main Jupyter Notebook).

Run the Training Block (Step 1) to generate the following files in your directory:
best_class_model.pkl
best_reg_model.pk
scaler.pkl
model_columns.pkl
shap_background.pkl
(Plus metric and image files)

### 6. Configure API Key
 Open the app.py file.
 Locate Line 30 (approx):
Python
gemini_key = "PASTE_YOUR_GEMINI_API_KEY_HERE"
Replace the placeholder text with your actual Google Gemini API key.

### 7. Run the Application
Bash
streamlit run app.py
The application will open in your default browser at http://localhost:8501.

## 📂 Project Structure
Plaintext
CreditIQ-AI-Underwriter/
├── app.py                   # Main Streamlit Application (Frontend & Logic)
├── credit_dataset.csv       # Dataset used for training
├── Training_Notebook.ipynb  # Jupyter Notebook for EDA & Model Training
├── requirements.txt         # List of Python dependencies
├── .gitignore               # Files to ignore (e.g., secrets, virtual env)
├── README.md                # Project Documentation
├── *.pkl                    # Saved Model Artifacts (Generated after training)
└── *.png                    # Dashboard Visualizations (Generated after training)

## 📊 Model Performance
  Classification Accuracy: ~80% (Logistic Regression)

  Regression R2 Score: ~0.90 (Neural Network)

  AUC Score: ~0.85

## 🤝 Contributing
  Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📜 License
  This project is licensed under the MIT License.
