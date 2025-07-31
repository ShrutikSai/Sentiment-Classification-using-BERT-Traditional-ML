# 🧾 Sentiment Classification using BERT and Traditional ML

An end-to-end sentiment analysis pipeline built on 42K+ comments utilizing BERT for deep learning and traditional machine learning models. Achieved an accuracy of 87% and an F1 score of 0.86. The project includes preprocessing with SpaCy/NLTK, TF-IDF vectorization, and GPU-accelerated mixed-precision training.

## 📌 Features

- ✅ Classifies sentiments of comments into positive, negative, and neutral categories  
- 🔍 Utilizes BERT for deep learning and traditional ML models for baseline comparisons  
- 📊 Achieves 87% accuracy and F1 score of 0.86 on the test dataset  
- ⚙️ Comprehensive pipeline: Data Collection → Preprocessing → Vectorization → Model Training → Evaluation  
- 🧩 Modular design for easy integration of new models and techniques  
- 📈 Visualizes results using confusion matrices and class distribution plots  

---

## 🚀 Tech Stack

| Layer              | Tools / Libraries                          |
|--------------------|---------------------------------------------|
| **Language**       | Python 3.10                                 |
| **Deep Learning**  | BERT (Hugging Face Transformers)            |
| **NLP Libraries**  | SpaCy, NLTK                                 |
| **Vectorization**  | TF-IDF                                      |
| **Deployment**     | Localhost (can be containerized via Docker) |
| **Visualization**  | Matplotlib, Seaborn                         |

---

## 📂 Project Structure

```
sentiment-classification/
│
├── data/                  # Dataset files
├── models/                # Saved ML/BERT models
├── notebooks/             # Jupyter notebooks for development & testing
├── scripts/               # Modular Python scripts for pipeline steps
├── utils/                 # Helper functions (e.g., preprocessor, tokenizer)
├── requirements.txt       # All dependencies
└── app.py                 # Main runner script
```

---

## 📊 Dataset

The project uses a **sentiment analysis dataset** available on Kaggle. You can download it from the following link:

🔗 [Sentiment Analysis Dataset – Kaggle](https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset)

> **Note**: Ensure to check the dataset link for availability.

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/Sentiment-Classification-using-BERT-Traditional-ML.git
cd Sentiment-Classification-using-BERT-Traditional-ML

# Install dependencies
pip install -r requirements.txt

# Run main pipeline
python app.py
```

---

## 📈 Results

| Metric                  | Value     |
|-------------------------|-----------|
| Accuracy                | 87%       |
| F1 Score                | 0.86      |
| Avg. Processing Time    | < 0.5s/comment |

---

## 🤝 Contributing

Feel free to open issues or submit pull requests to enhance the model performance or add new features!

---

## 📄 License

This project is licensed under the MIT License. See LICENSE for more details.
