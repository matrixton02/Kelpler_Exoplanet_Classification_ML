# KNN Distance Metric Comparison for Exoplanet Detection  
**Dataset:** NASA Exoplanet Archive – Kepler Candidates  
**Task:** Binary Classification (Confirmed Planet vs False Positive)  
**Models Implemented:** KNN from scratch with multiple distance metrics

---

## 📌 Project Overview

This project implements the **K-Nearest Neighbors (KNN)** and **Logistic Regression** algorithm **from scratch in Python** and evaluates how different **distance metrics** affect classification accuracy on a real scientific dataset:  
NASA’s Kepler exoplanet candidate catalog.

The goal is to understand how different classification model behave for the same dataset we also used different geometric interpretations of distance to improve exoplanet detection performance.

---

## 📊 Dataset Description

We use the **Kepler “cumulative” candidate catalog** from the NASA Exoplanet Archive.
###Link: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results
### Features used:
- `koi_period` – Orbital period  
- `koi_depth` – Transit depth  
- `koi_duration` – Transit duration  
- `koi_prad` – Planet radius (Earth radii)  
- `koi_teq` – Equilibrium temperature  
- `koi_insol` – Insolation flux  
- `koi_steff` – Stellar temperature  
- `koi_srad` – Stellar radius  
- `koi_smass` – Stellar mass  

### Target label:
- **1** → Confirmed Planet  
- **0** → False Positive  

Only rows with valid Kepler dispositions were included.

---

## 🧠 Distance Metrics Implemented

### 1️⃣ **Euclidean Distance**
\[
d = \sqrt{\sum (x_i - y_i)^2}
\]

### 2️⃣ **Manhattan Distance**
\[
d = \sum |x_i - y_i|
\]

### 3️⃣ **Mahalanobis Distance**
\[
d = \sqrt{(x-y)^T S^{-1}(x-y)}
\]
Accounts for **feature correlations** using the covariance matrix.

### 4️⃣ **RBF Kernel Distance**
\[
sim = e^{-\gamma \|x-y\|^2}, \quad d = 1 - sim
\]
A nonlinear distance metric.

### 5️⃣ **RBF Normalized**
Scaled variant:

\[
d = \frac{\|x-y\|^2}{n_{\text{features}}}
\]

---

## 🧪 Experimental Results

| Distance Metric | Accuracy | Notes |
|------------------|----------|------------------------------|
| **Euclidean** | **78.5%** | Baseline distance metric |
| **Manhattan** | **78.5%** | Same neighbor ordering as Euclidean after scaling |
| **Mahalanobis** | **84.5%** | Best performance; accounts for astrophysical correlations |
| **RBF (γ=0.01)** | **78.03%** | Behaves similarly to Euclidean (small γ → linear) |
| **RBF (γ=0.1)** | **64.4%** | γ too large → nonlinear distortion hurts performance |
| **RBF (γ=0.0001)** | **78.3%** | Again approximates Euclidean |
| **Logistic Regression (baseline)** | **82.5%** | Shows approximate linear separability |

---

## 📌 Scientific Insights

### 🔹 1. **Kepler features are moderately linearly separable**
This is why logistic regression performs well (82.5%).

### 🔹 2. **Euclidean and Manhattan give identical accuracy**
After feature standardization, both metrics induce nearly identical neighborhood structure.

### 🔹 3. **Mahalanobis is the best performer (84.5%)**
Because:
- transit features are **correlated**  
- Mahalanobis "whitens" the feature space  
- improving nearest-neighbor geometry  

This is expected in astrophysical datasets where stellar parameters influence multiple planetary observables.

### 🔹 4. **RBF kernel does NOT improve performance**
The dataset does not exhibit strong nonlinear boundaries.

With large γ, RBF collapses distances and performance drops significantly (~64%).

With small γ, RBF behaves like Euclidean → 78%.

You can test with different γ yourself and see the difference.
---

## 📁 Repository Structure
```
├── knn.py # KNN implementation with all distance metrics
├── preprocess.py # Data loading and cleaning for Kepler dataset
├── Kepler_Identification_KNN.py # Using KNN on the Kepler dataset using different distance fucntions
├── Kepler_identification_Lg.py # Using Logistic regression on the Kepler dataset
├── Knn_result.png # Accuracy bar graph for KNN representing the accuracy of different dataset fucnctions
├── Lg_result.png # Shows the ROC_AOC curver and cost fucntion mapping for the logistic regression model
├── LICENSE # MIT license for the project
└── README.md
```
---
## 🚀 How to Run
If you want to runt he logistic regression model:
```
python Kepler_Identification_Lg.py
```
If you want to run the KNN model:
```
python Kepler_Identification_KNN.py
```
## 🏁 Conclusion

This project demonstrates how distance metrics fundamentally change the geometry of KNN, especially in scientific datasets.

Key takeaway:
Exoplanet detection from Kepler features is not strongly non-linear,but is influenced by correlated astrophysical parameters.
When features are correlated (as in astrophysics), Mahalanobis distance is superior.

## 📞 Contact
Feel free to explore the repo and message me about improvements or questions!
Email:yashasvi21022005@gmail.com
LinkedIn:https://www.linkedin.com/in/yashasvi-kumar-tiwari/


