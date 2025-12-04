# Telco Customer Churn Prediction

A comprehensive machine learning project analyzing customer churn patterns in a telecommunications company using Logistic Regression. This portfolio project demonstrates end-to-end data science workflow from data preprocessing to actionable business insights.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Project Overview

### Business Problem

Customer churn is a critical challenge in the telecommunications industry, with acquiring new customers costing 5-25x more than retaining existing ones. This project builds a predictive model to:

- Identify customers at high risk of churning
- Understand key drivers of customer attrition
- Enable proactive retention strategies

### Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: 7,043 customers
- **Features**: 20 (demographics, services, billing)
- **Target**: Binary (Churn: Yes/No)
- **Class Distribution**: 73.4% retained, 26.6% churned

## 🎯 Key Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 80.45% | Overall correctness of predictions |
| **Precision** | 65.05% | Of predicted churners, 65% actually churned |
| **Recall** | 57.22% | Identified 57% of actual churners |
| **F1-Score** | 60.88% | Balanced performance metric |

### Confusion Matrix

```
                Predicted
             No Churn  |  Churn
Actual
No Churn      918      |  115
Churn         160      |  214
```

- **True Negatives (918)**: Correctly identified loyal customers
- **True Positives (214)**: Correctly identified churners
- **False Negatives (160)**: Missed at-risk customers ⚠️
- **False Positives (115)**: False alarms (wasted retention efforts)

## 🔍 Key Insights

### Top Churn Drivers (Features to Address)

1. **Fiber Optic Internet** (+1.12) - Highest churn risk
   - *Action*: Investigate service quality and competitive pricing

2. **Total Charges** (+0.64) - Higher spending correlates with churn
   - *Action*: Review pricing strategy and perceived value

3. **Electronic Check Payment** (+0.39) - Less committed payment method
   - *Action*: Incentivize automatic payment enrollment

4. **Streaming TV** (+0.37) - Possible competitor threat
   - *Action*: Bundle optimization and content partnerships

### Top Protective Factors (Features to Promote)

1. **Two-Year Contracts** (-1.35) - Strongest churn prevention
   - *Action*: Offer incentives for contract upgrades (e.g., 20% discount)

2. **Customer Tenure** (-1.36) - Loyalty builds over time
   - *Action*: Focus on 90-day onboarding program for new customers

3. **One-Year Contracts** (-0.74) - Moderate protection
   - *Action*: Use as stepping stone to two-year commitments

4. **Online Security Service** (-0.37) - Value-add retention tool
   - *Action*: Promote security bundles in marketing

## 🛠️ Technical Implementation

### Technologies Used

- **Python 3.8+**: Core programming language
- **pandas & NumPy**: Data manipulation and analysis
- **scikit-learn**: Machine learning pipeline
- **matplotlib & seaborn**: Data visualization
- **KaggleHub**: Dataset loading

### Methodology

#### 1. Data Preprocessing

- **Data Cleaning**: Removed 11 rows (0.16%) with invalid `TotalCharges`
- **Feature Engineering**:
  - Binary encoding for 7 Yes/No features
  - One-hot encoding for 10 multi-class categorical features
  - Created 21 dummy variables
- **Final Dataset**: 7,032 rows × 31 columns (30 features + 1 target)

#### 2. Model Pipeline

**Train/Test Split**: 80/20 with stratification

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Feature Scaling** (fitted on training data only)

```python
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
```

**Logistic Regression**

```python
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
```

#### 3. Avoiding Data Leakage

✅ **Correct approach implemented**:

- Split data **before** scaling
- Fitted scaler on training set only
- Transformed test set using training parameters

#### 4. Model Performance

- **Training Time**: 0.11 seconds (extremely fast)
- **Training Accuracy**: 80.48%
- **Test Accuracy**: 80.45% (good generalization, no overfitting)

## 📈 Business Impact

### Risk Segmentation Strategy

Using probability thresholds to prioritize interventions:

| Risk Level | Probability | Customers | Action | Expected ROI |
|-----------|-------------|-----------|--------|--------------|
| **High** | ≥ 0.70 | ~180 (12.8%) | Immediate intervention: 30% discount + upgrade | Retain 60% = $120K/year |
| **Medium** | 0.40-0.70 | ~320 (22.8%) | Proactive outreach: 15% discount | Retain 30% = $45K/year |
| **Low** | < 0.40 | ~906 (64.4%) | Standard monitoring | Maintain satisfaction |

### Hypothetical Cost-Benefit Analysis

*Assumptions*:
- Average customer lifetime value: $1,000
- Retention campaign cost: $100/customer

**Results**:
- **Cost of False Positives** (wasted campaigns): $11,500
- **Cost of False Negatives** (lost customers): $160,000
- **Value of True Positives** (retained customers): $192,600
- **Net Benefit**: $21,100

→ **Model provides positive ROI** and justifies deployment

## 📁 Project Structure

```
telco-churn-analysis/
│
├── telco_churn_analysis.ipynb      # Main Jupyter notebook
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── images/
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── models/
    └── logistic_regression_model.pkl
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
```

### Running the Notebook

**Clone repository**

```bash
git clone https://github.com/yourusername/telco-churn-analysis.git
cd telco-churn-analysis
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Open Jupyter notebook**

```bash
jupyter notebook telco_churn_analysis.ipynb
```

### Quick Start (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `telco_churn_analysis.ipynb`
3. Run all cells (dataset loads automatically via KaggleHub)

## 🔮 Future Improvements

### Model Enhancements

- [ ] Test ensemble methods (Random Forest, XGBoost, LightGBM)
- [ ] Hyperparameter tuning via GridSearchCV
- [ ] Implement SMOTE for class imbalance handling
- [ ] Add SHAP values for explainability

### Feature Engineering

- [ ] Create interaction features (e.g., tenure × contract type)
- [ ] Engineer customer lifetime value (CLV) metric
- [ ] Add time-series features (trend analysis)

### Deployment

- [ ] Build Flask API for real-time predictions
- [ ] Create Streamlit dashboard for stakeholders
- [ ] Implement A/B testing framework
- [ ] Set up MLOps pipeline (MLflow tracking)

## 📚 Lessons Learned

1. **Class Imbalance Matters**: Stratified sampling ensures representative splits
2. **Data Leakage Prevention**: Always scale after splitting
3. **Business Context > Accuracy**: Precision-recall trade-offs depend on cost structure
4. **Interpretability is Key**: Feature importance drives actionable insights

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Portfolio: [yourportfolio.com](https://yourportfolio.com)

## 🙏 Acknowledgments

- Dataset: [Kaggle - BlastChar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Methodology: CRISP-DM framework
- Inspiration: IBM Watson Analytics sample datasets

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
