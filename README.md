# spam-detection-ml
A Streamlit-based SMS spam detection system that uses machine learning and NLP to classify messages as spam or ham with support for single and batch predictions.
```markdown
# 🛡️ Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](#)

A powerful, interactive Machine Learning web application that predicts whether an SMS or Email message is **Spam** or **Ham (Legitimate)**. Built with Python and Streamlit, this app offers real-time single message analysis, batch CSV processing, and an interactive data dashboard.

### 🔗 Links
* **🟢 Live Demo:** [Click here to view the live app](https://spam-predictor.streamlit.app/)
* **📂 GitHub Repository:** [Thatipramod/spam-detection-ml](https://github.com/Thatipramod/spam-detection-ml)

---

## 📸 Project Screenshots

| 🏠 Home / Predictor | 📂 Batch Prediction |
| :---: | :---: |
| ![Spam Predictor](images/predictor.png) | ![Batch Prediction](images/batch.png) |

| 📊 Analytics | 📈 Dashboard |
| :---: | :---: |
| ![Analytics](images/analytics.png) | ![Dashboard](images/dashboard.png) |

---

## 🚀 Key Features

* **📩 Real-time Prediction:** Paste any message or email and instantly find out if it's spam, complete with a confidence score.
* **📂 Batch Processing:** Upload a CSV file of messages and let the model classify all of them at once.
* **📊 Visual Analytics:** Explore the dataset through interactive charts, word clouds, and distribution graphs.
* **📋 Export Results:** Download your batch prediction results as a clean CSV file.
* **⚡ Fast & Lightweight:** Powered by an optimized Logistic Regression model and TF-IDF vectorization.

---

## 🤖 Machine Learning Details

**Dataset Used:** SMS Spam Collection Dataset (5,572 messages)
* ✅ **Ham:** ~86.6%
* 🚨 **Spam:** ~13.4%

**Algorithm Selection:**
During development, three algorithms were tested: *Naive Bayes*, *Random Forest*, and *Logistic Regression*. 
**Logistic Regression** was chosen as the final model because it provided the best balance between overall accuracy, processing speed, and precision in identifying spam.

### ⚙️ How it Works (Under the Hood)
1. **Load Dataset:** Raw text messages are ingested.
2. **Clean Text:** Punctuation is removed, URLs are stripped, and text is standardized.
3. **TF-IDF Vectorization:** Words are converted into numerical importance scores.
4. **Classification:** The Logistic Regression model evaluates the vector.
5. **Output:** The app displays the Spam/Ham result and confidence metric.

---

## 📚 Tech Stack & Requirements

To run this project, you need the following Python libraries installed:

**`requirements.txt`**
```text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
wordcloud

```

---

## 🛠️ Installation & Local Setup

Want to run this app on your local machine? Follow these steps:

**1. Clone the repository**

```bash
git clone [https://github.com/Thatipramod/spam-detection-ml.git](https://github.com/Thatipramod/spam-detection-ml.git)
cd spam-detection-ml

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Train the model (Optional)**
*If you need to generate a fresh `model.pkl` file, run the training script:*

```bash
python train_model.py

```

**4. Run the Streamlit app**

```bash
streamlit run app.py

```

---

## 🔮 Future Enhancements

* [ ] Add explicit support for extensive Email Spam Detection.
* [ ] Build a REST API for real-time predictions outside the web app.
* [ ] Upgrade to Deep Learning models (LSTM / BERT) for higher context understanding.
* [ ] Introduce Multi-language spam detection.
* [ ] Deploy seamlessly via Docker.

---

## 👨‍💻 Developer

**Thati Pramod**

* **Role:** Full-stack Developer & ML Engineer
* **Language:** Python
* **Framework:** Streamlit, Scikit-Learn

*Thank you for checking out the Spam Detection System! If you found this helpful, feel free to give the repository a ⭐.*

```

```
