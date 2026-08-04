# Flipkart CSAT Machine Learning Classification

[](https://www.google.com/search?q=https://github.com/BATTI-CHANDAN-SINGH/Flipkart-CSAT-Ml-Classification)[cite: 3]

---

## **Project Overview**

Flipkart handles a high volume of customer interactions across multiple issue categories, including orders, payments, returns, delivery delays, and product quality[cite: 3].

The primary goal of this project is to build a Machine Learning Classification model that predicts customer dissatisfaction in advance using operational, order, agent, and interaction metrics[cite: 3]. By proactively identifying unsatisfied customer tickets, the system helps prioritize high-risk issues, optimize agent handling efficiency, reduce churn, and improve overall customer retention[cite: 3].

---

## **Key Features & Objectives**

* **Target Variable:** Binary classification of customer satisfaction (`Satisfied` vs. `Unsatisfied`) based on CSAT scores[cite: 3].
* **Feature Engineering:** Extraction of interaction metrics such as Response Duration, Order-to-Issue Lag, Survey Response Delay, Issue Hour/Day indicators, and Agent Experience Tiers[cite: 3].
* **Text Preprocessing:** Cleaning and vectorizing customer text remarks using lowecasing, stopword removal, lemmatization, and TF-IDF / Count Vectorizer[cite: 3].
* **Machine Learning Pipelines:** Training and evaluation of multiple classification models, including Logistic Regression, Decision Trees, Random Forests, and XGBoost[cite: 3].
* **Optimization & Evaluation:** Hyperparameter tuning using GridSearch/RandomSearch and Stratified K-Fold Cross-Validation, prioritizing **Recall** and **F1-Score** to minimize missed dissatisfied cases[cite: 3].

---

## **Project Structure**

```text
Flipkart-CSAT-Ml-Classification/
│
├── data/
│   └── flipkart_csat_data.csv          # Dataset containing customer interaction metrics
│
├── notebooks/
│   └── Flipkart_CSAT_ML_Pipeline.ipynb  # Exploratory Data Analysis & Modeling Notebook
│
├── README.md                           # Project Documentation
└── requirements.txt                    # Project Dependencies

```

---

## **Workflow & Methodology**

```
 ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
 │  1. Know Your Data   │ ──> │  2. Data Wrangling   │ ──> │ 3. Exploratory Data  │
 │ (Nulls, Duplicates)  │     │ (Cleaning & Dates)   │     │   Analysis (UBM)     │
 └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                       │
 ┌──────────────────────┐     ┌──────────────────────┐                 ▼
 │ 6. Model Evaluation  │ <── │ 5. Machine Learning  │ <── ┌──────────────────────┐
 │  & Interpretability  │     │ (LR, RF, XGBoost)    │     │ 4. Feature Engg. &   │
 └──────────────────────┘     └──────────────────────┘     │   Text Preprocessing │
                                                           └──────────────────────┘
```[cite: 3]

### **1. Data Preprocessing & Cleaning**
* Checked for null values, duplicates, and data type inconsistencies[cite: 3].
* Applied datetime conversions to engineer temporal features (e.g., interaction lag, time-of-day flags)[cite: 3].

### **2. Exploratory Data Analysis (EDA - UBM Rule)**
* **Univariate:** Class distribution analysis of CSAT targets, issue categories, and support channels[cite: 3].
* **Bivariate & Multivariate:** Analyzed correlation between agent experience vs. CSAT, handling duration vs. satisfaction, and channel-wise satisfaction trends[cite: 3].

### **3. Modeling & Evaluation Metrics**
* Evaluated models on **Recall**, **F1-Score**, and **ROC-AUC**[cite: 3].
* Focused on minimizing **False Negatives** to ensure at-risk unsatisfied customers are accurately flagged before escalation[cite: 3].

---

## **Tech Stack**

* **Language:** Python[cite: 3]
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, NLTK, Matplotlib, Seaborn[cite: 3]
* **Development Environment:** Jupyter Notebook / Google Colab[cite: 3]

---

## **Installation & Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BATTI-CHANDAN-SINGH/Flipkart-CSAT-Ml-Classification.git
   cd Flipkart-CSAT-Ml-Classification
   ```[cite: 3]

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt

```

3. **Run the Notebook:**
Open `notebooks/Flipkart_CSAT_ML_Pipeline.ipynb` in Jupyter Notebook or VS Code to execute the pipeline[cite: 3].
