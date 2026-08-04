# 🛡️ Spam Detection ML

A **Streamlit-based SMS Spam Detection System** that uses **Machine Learning** and **Natural Language Processing (NLP)** to classify messages as **Spam** or **Ham**, with support for both single-message and batch predictions.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Spam Detection ML is a machine learning web application that classifies SMS messages as **Spam** or **Ham**. Built with **Python**, **Streamlit**, and **Scikit-learn**, the application provides an intuitive interface for real-time predictions, batch processing, and data analytics.

---

## 🚀 Features

- 📩 Real-time SMS spam prediction
- 📂 Batch prediction using CSV files
- 📊 Interactive analytics dashboard
- 📈 Dataset visualization and statistics
- 🤖 Machine Learning-based classification
- ⚡ Fast and lightweight Streamlit interface

---

## 🖥️ Application Pages

- 🏠 Home
- 📩 Spam Predictor
- 📂 Batch Prediction
- 📊 Analytics
- 📈 Dashboard
- ℹ️ About

---

## 🧠 Machine Learning Pipeline

1. Data preprocessing
2. Text cleaning
3. Tokenization
4. TF-IDF Vectorization
5. Logistic Regression Classification
6. Spam/Ham Prediction

### Model

- **Algorithm:** Logistic Regression
- **Vectorizer:** TF-IDF
- **Language:** Python
- **Framework:** Scikit-learn

---

## 📂 Project Structure

```text
spam-detection-ml/
│
├── app.py
├── requirements.txt
├── train_model.py
├── utils.py
├── style.css
├── spam.csv
│
├── dataset/
├── pages/
│   ├── About.py
│   ├── Analytics.py
│   ├── Batch_Prediction.py
│   ├── Dashboard.py
│   └── Spam_Predictor.py
│
├── plots/
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- NLTK

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/Thatipramod/spam-detection-ml.git
cd spam-detection-ml
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📷 Screenshots

> Add your application screenshots here.

```
screenshots/
├── home.png
├── predictor.png
├── batch_prediction.png
├── analytics.png
└── dashboard.png
```

Example:

```markdown
![Home](screenshots/home.png)

![Spam Predictor](screenshots/predictor.png)

![Analytics](screenshots/analytics.png)
```

---

## 📊 Dataset

- SMS Spam Collection Dataset
- Total Messages: **5,572**
- Ham Messages: **4,825**
- Spam Messages: **747**

---

## 🔮 Future Improvements

- Email spam detection
- Deep Learning models (LSTM/BERT)
- REST API integration
- Multi-language support
- Docker deployment
- Cloud database integration

---

## 🌐 Live Demo

**Streamlit App**

https://spamguard.streamlit.app

---

## 📁 GitHub Repository

https://github.com/Thatipramod/spam-detection-ml

---

## 👨‍💻 Author

**Thati Pramod**

GitHub: https://github.com/Thatipramod

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub.

It helps others discover the project and motivates future improvements.
