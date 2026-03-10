# 🍄 Mushroom Classification

A web application that classifies mushrooms as **edible** or **poisonous** using a **Gaussian Naive Bayes** machine learning model, built with [Streamlit](https://streamlit.io/).

## ✨ Features

- **🔍 Mushroom Prediction** — Select 21 physical characteristics of a mushroom and get an instant edible/poisonous classification with confidence score.
- **📊 Model Performance** — View evaluation metrics (Accuracy, Precision, Recall, F1-Score, MCC, AUC-ROC) with an interactive radar chart.
- **🟩 Confusion Matrix** — Visualize prediction results with a heatmap and TP/TN/FP/FN breakdown.
- **📈 ROC Curve** — Inspect the Receiver Operating Characteristic curve with AUC score.

## 📸 Screenshots

> _Run the app locally to see the Forest Green + Gold themed UI._

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML Model | Gaussian Naive Bayes (scikit-learn) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dataset | [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) |

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:Birdie-1/mushroom-classification.git
   cd mushroom-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
mushroom-classification/
├── app.py              # Main Streamlit application (UI + model training)
├── mushrooms.csv       # UCI Mushroom Dataset
├── requirements.txt    # Python dependencies
└── README.md
```

## 📊 Dataset

The app uses the **UCI Mushroom Dataset** containing **8,124 samples** of mushrooms with **22 categorical features** (e.g., cap shape, odor, gill color, habitat). The `veil-type` column is dropped during preprocessing as it contains only a single unique value.

- **Classes**: Edible (`e`) / Poisonous (`p`)
- **Split**: 80% training / 20% testing (stratified)

## 🤖 Model

The app trains a **Gaussian Naive Bayes** classifier on the label-encoded features. All training happens at startup and is cached by Streamlit for fast subsequent loads.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Correct positive predictions / Total positive predictions |
| Recall | Correct positive predictions / Total actual positives |
| F1-Score | Harmonic mean of Precision & Recall |
| MCC | Matthews Correlation Coefficient (−1 to 1) |
| AUC-ROC | Area under the ROC curve |

## 📝 License

This project is for educational purposes.
