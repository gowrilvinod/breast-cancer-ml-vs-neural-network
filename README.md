# Breast Cancer Classification: Classical ML vs Neural Network

This project compares classical machine learning models (Logistic Regression and Random Forest) with a Neural Network implemented in PyTorch for binary breast cancer classification.

The goal is to evaluate whether deep learning significantly outperforms traditional machine learning models for structured biomedical tabular data.

---

##  Dataset

- Breast Cancer Wisconsin dataset (scikit-learn)
- 569 samples
- 30 numerical features
- Binary classification:
  - 0 = Malignant
  - 1 = Benign

The dataset shows mild class imbalance but does not require resampling.

---

## 🔎 Exploratory Data Analysis

- Class distribution visualization
- Correlation heatmap (strong multicollinearity observed)
- PCA visualization (clear class separability along PC1)

---

##  Models Implemented

###  Logistic Regression
- L2 regularization (default)
- L1 vs L2 regularization comparison
- Scaled features using StandardScaler

###  Random Forest
- 100 decision trees
- Captures nonlinear relationships
- Feature importance analysis performed

###  Neural Network (PyTorch)
Architecture:
- Input layer: 30 features
- Hidden layer: 16 neurons (ReLU)
- Output layer: 1 neuron
- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Trained for 100 epochs

---

##  Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- AUC (ROC Curve)

Additionally:
- 5-fold Cross-Validation was performed for robustness analysis.

---

## Results Summary

- Logistic Regression** achieved approximately 97.4% accuracy with an AUC of ~0.997, demonstrating the most stable and consistent performance across evaluations.

- Random Forest** achieved approximately 96.5% accuracy with an AUC of ~0.995, performing competitively while offering useful feature importance insights.

- Neural Network** achieved approximately 96.5% accuracy with an AUC of ~0.997, showing comparable performance but without a significant improvement over classical models.

Cross-validation showed that Logistic Regression achieved the most stable performance (mean CV accuracy ≈ 98%).

---

## Key Insights

- The dataset is highly structured and likely close to linearly separable.
- Logistic Regression performs exceptionally well on tabular biomedical data.
- Neural Networks do not significantly outperform classical ML in this setting.
- Random Forest provides strong interpretability through feature importance analysis.

---

##  Conclusion

For structured tabular datasets with moderate size and strong feature engineering, classical machine learning models remain highly effective and competitive with deep learning approaches.

Model complexity alone does not guarantee superior performance.

---

##  Technologies Used

- Python
- scikit-learn
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

##  How to Run

1. Clone the repository
2. Install required packages:
   pip install -r requirements.txt
3.Run the Jupyter Notebook
