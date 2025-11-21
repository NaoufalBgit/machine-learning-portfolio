# Handwritten Digit Recognition — MNIST

This project applies several machine learning techniques to the MNIST dataset, a benchmark dataset containing **70,000 handwritten digit images** (28×28 grayscale).  
The aim is to compare different models, analyze classification errors, and combine predictions to improve performance.

## 📁 Notebooks

### 🧪 1. KNN Classification  
**File:** `KNN_MNIST_Classification.ipynb`

This notebook includes:
- Loading and preprocessing the MNIST dataset  
- Stratified cross-validation to select the best hyperparameters of `KNeighborsClassifier`  
  - `n_neighbors`: 3, 7, 15  
  - `weights`: `uniform`, `distance`  
  - `metric`: `cosine`, `euclidean`, `manhattan`  
- Training the final KNN classifier with the optimal parameters  
- Evaluation on the test set (accuracy and confusion matrix)  
- Analysis of the confusion matrix to identify ambiguous digits  
- Visualization of misclassified images  

---

### 🧪 2. Logistic Regression (OvA + OvO)  
**File:** `LogisticRegression_MNIST_OvA_OvO.ipynb`

This notebook develops:
- **10 One-vs-All (OvA)** logistic regression classifiers (one per digit)  
- **45 One-vs-One (OvO)** classifiers for all digit pairs  
- A **conflict-resolution function** combining OvA and OvO predictions  
  - Case where no OvA classifier fires  
  - Case where several OvA classifiers fire (A, B, C) with cyclic OvO outcomes  
- Evaluation using accuracy, confusion matrix, and error inspection  
- Visualization of hard-to-classify digits  

---

## 🔄 Prediction Combination

Two meta-models are explored to combine predictions:

### Decision Tree  
Inputs:
- KNN prediction  
- 10 OvA predictions  
- 45 OvO predictions  

### Logistic Regression  
Same inputs, with KNN encoded as a one-hot vector (size 10).

Both models are trained and evaluated to assess whether combining predictions improves classification accuracy.

---

## 🚀 Further Experiments

Additional enhancements explored include:
- Data augmentation by shifting white-pixel rows (up, down, left, right)  
- Retraining models on the extended dataset  
- Comparing performance gains across all classifiers  

---

## 📊 Dataset Information

The dataset is loaded via:

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
