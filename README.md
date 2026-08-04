# 🛒 Flipkart E-Commerce Product Categorization

An end-to-end Natural Language Processing (NLP) and Machine Learning classification pipeline built to automate product category prediction on Flipkart's e-commerce dataset. By parsing unstructured text features—such as product titles and descriptions—this project extracts key textual attributes to accurately assign items to primary categories.

---

## 📌 Project Overview

E-commerce catalogs handle vast amounts of product listings across thousands of sub-tree categories. Manual categorization leads to search friction, indexing delays, and misclassified items.

This project automates product classification by extracting textual features from product metadata, preprocessing the corpus, and evaluating multi-class Machine Learning models to optimize prediction accuracy across top product classes.

### Key Objectives
* Parse and clean raw product descriptions, removing noise, formatting tags, and custom stopwords.
* Map deep hierarchical category trees (`Product Category Tree`) down to distinct primary root categories.
* Convert unstructured text into machine-readable numeric representations using **TF-IDF Vectorization** (Unigrams & Bigrams).
* Train and benchmark multiple standard classifiers (Linear Support Vector Classifier, Naive Bayes, Random Forest).
* Evaluate performance using stratified splits, accuracy scores, macro F1 metrics, and confusion matrices.

---

## 📁 Repository Structure


```

├── dataset/
│   ├── raw_flipkart_data.csv          # Raw Flipkart e-commerce dataset
│   └── cleaned_flipkart_data.csv      # Preprocessed features and extracted primary categories
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb # Visualization, class counts, and text profiling
│   ├── 02_Data_Preprocessing.ipynb    # Stopword removal, lemmatization, and TF-IDF extraction
│   └── 03_Model_Training_Evaluation.ipynb # Machine Learning classifiers and model benchmarking
├── src/
│   ├── preprocessing.py               # Text cleaning functions and vectorization utilities
│   └── train.py                       # Automated training and evaluation scripts
├── models/
│   └── vectorizer_and_model.pkl       # Saved TF-IDF vectorizer and trained best-performing classifier
├── requirements.txt                   # Dependency requirements
└── README.md                          # Project documentation

```

---

## 📊 Exploratory Data Analysis & Data Cleaning

### Data Insights & Challenges
1. **Category Normalization:** Raw metadata contains nested strings (e.g., `["Clothing >> Women's Clothing >> Western Wear..."]`). These trees were parsed to extract the primary top-level root category.
2. **Missing & Noisy Data:** Missing description values were handled, and rare categories with insufficient sample representation were filtered out to prevent severe class distribution skew.
3. **Primary Classes Covered:** The dataset focuses on major e-commerce categories including *Clothing*, *Jewellery*, *Sports & Fitness*, *Electronics*, *Babycare*, *Home Furnishing & Kitchen*, *Footwear*, and *Tools & Hardware*.

---

## ⚙️ Text Preprocessing & Feature Engineering


```

+---------------------+     +-----------------------+     +------------------------+
|  Raw Product Text   | --> | Data Cleaning         | --> | TF-IDF Vectorization   |
|  (Title/Description)|     | (Lower, Stopwords,    |     | (Word Unigrams/Bigrams |
|                     |     |  Lemmatization)       |     |  Character N-Grams)    |
+---------------------+     +-----------------------+     +------------------------+
|
v
+---------------------+     +-----------------------+     +------------------------+
| Saved Pipeline      | <-- | Model Benchmarking    | <-- | Stratified Train/Test  |
| (.pkl Artifacts)    |     | (Linear SVC, MNB,     |     | (70:30 or 80:20 Split) |
|                     |     |  Random Forest)       |     |                        |
+---------------------+     +-----------------------+     +------------------------+

```

1. **Text Normalization:** Stripped special characters, digits, punctuation, and URLs.
2. **Tokenization & Lemmatization:** Converted tokens to lower-case and reduced words to their root forms using WordNet Lemmatizer.
3. **TF-IDF Feature Extraction:** Applied $n$-gram TF-IDF representations (Unigrams + Bigrams) to capture single words and contextual keyphrases.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Natural Language Processing:** `nltk`
* **Machine Learning Pipelines:** `scikit-learn`
* **Environment:** Jupyter Notebook / Google Colab

---

## 📈 Model Performance & Evaluation

The preprocessed text features were evaluated across standard machine learning classifiers using stratified train-test splits:

| Machine Learning Model | Feature Vector Representation | Overall Accuracy |
| :--- | :--- | :---: |
| **Multinomial Naive Bayes** | Word TF-IDF (Unigrams + Bigrams) | ~95.5% |
| **Random Forest Classifier** | Word TF-IDF (Unigrams + Bigrams) | ~98.3% |
| **Linear Support Vector Classifier (Linear SVC)** | Word TF-IDF (Unigrams + Bigrams) | **~98.8%** |

> **Key Takeaway:** The Linear SVC model combined with word-level TF-IDF feature extraction provided the highest precision and overall accuracy across primary product categories.

---

## 🚀 How to Set Up & Execute

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Flipkart-ML-Classification.git](https://github.com/your-username/Flipkart-ML-Classification.git)
cd Flipkart-ML-Classification

```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run Notebooks

Launch Jupyter Notebook to run through data preprocessing and model evaluation step-by-step:

```bash
jupyter notebook notebooks/01_Exploratory_Data_Analysis.ipynb

```

---

## 🔮 Future Improvements

* **Transformer Deep Learning:** Experiment with fine-tuning transformer models like BERT/DistilBERT for multi-label product classification.
* **Multimodal Data Fusion:** Incorporate image feature vectors alongside text representations to build a hybrid classification engine.
* **REST API Deployment:** Wrap the trained model and vectorizer in a FastAPI/Flask application for real-time predictions.
