---
title: Customer Churn Prediction
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app/app.py
pinned: false
---
# Customer Churn Prediction Project

This project is an end-to-end Machine Learning solution designed to predict customer churn based on their demographic information, account information, and service usage. It includes data exploration, preprocessing, model training, and a Streamlit web application for real-time predictions.

## Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── churn_model.pkl
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "Customer Churn Prediction Project"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web App

To start the Streamlit web application locally, run the following command from the root of the project:

```bash
streamlit run app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`. 

## Best Practices Followed
- **Modular Notebooks**: EDA, preprocessing, and training are separated into different notebooks for clarity.
- **Cleared Notebook Outputs**: All Jupyter notebook outputs are cleared before pushing to GitHub to prevent rendering errors and merge conflicts.
- **Streamlit Deployment**: Provides an interactive user interface for non-technical stakeholders to test the model.
