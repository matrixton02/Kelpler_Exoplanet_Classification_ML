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

## Distance Metrics Implemented
Given two points **x** and **y** in n-dimensional space:

### 1. Euclidean Distance
d(x, y) = sqrt( (x₁ - y₁)² + (x₂ - y₂)² + ... + (xₙ - yₙ)² )

### 2. Manhattan Distance
d(x, y) = |x₁ - y₁| + |x₂ - y₂| + ... + |xₙ - yₙ|

### 3. Mahalanobis Distance
d(x, y) = sqrt( (x - y)ᵀ · S⁻¹ · (x - y) )

where:
- (x - y) is the column vector of differences
- S is the covariance matrix of the features
- S⁻¹ is the inverse of S

### 4. RBF Kernel Distance
First define similarity using the RBF kernel:

sim(x, y) = exp( -γ · ||x - y||² )

Then convert to a distance:

d(x, y) = 1 - sim(x, y)

### 5. Normalized Squared Distance
d(x, y) = ( ||x - y||² ) / n_features
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
## 📊 Plots

### The Logistic regression plot

<img width="4500" height="1800" alt="image" src="https://github.com/user-attachments/assets/84f86b51-c0db-4c16-8d59-168f9e671362" />

### The KNN plot

<img width="3600" height="1800" alt="image" src="https://github.com/user-attachments/assets/86b104ed-420a-4f86-8db3-6b1819779fbc" />

## Logistic Regression Performance (Kepler Exoplanet Dataset)

### **Validation Set (80%)**
- **Accuracy:** 0.8179  
- **ROC–AUC:** 0.8879  

**Confusion Matrix**
|           |Predicted: Not Exoplanet	| Predicted: Exoplanet|
|-----------|-------------------------|---------------------|
| **Actual: Not Exoplanet** | 3011	| 654 |
| **Actual: Exoplanet**	| 413	| 1782

Accuracy: 0.8117

ROC–AUC: 0.8859

Confusion Matrix

|               | Predicted: Not Exoplanet | Predicted: Exoplanet |
|--------------|---------------------------|------------------------|
| **Actual: Not Exoplanet** | 736 | 181 |
| **Actual: Exoplanet**     | 95  | 454 |

## 📌 Scientific Insight

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
├── K_nearest_neighbour.py # KNN implementation with all distance metrics
├── Data_preprocessor.py # Data loading and cleaning for Kepler dataset
├── Kepler_Identification_KNN.py # Using KNN on the Kepler dataset using different distance fucntions
├── Kepler_identification_Lg.py # Using Logistic regression on the Kepler dataset
├── Knn_result.png # Accuracy bar graph for KNN representing the accuracy of different dataset fucnctions
├── Lg_result.png # Shows the ROC_AOC curver and cost fucntion mapping for the logistic regression model
├── LICENSE # MIT license for the project
└── README.md
```
---
## 🚀 How to Run
Download the dataset keep it int he same folder and then go to the Kepler_implementation_KNN/LG.py and under if __name__=="__main__" update the filepath variable to your dataset location then,

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

Email: yashasvi21022005@gmail.com

LinkedIn: https://www.linkedin.com/in/yashasvi-kumar-tiwari/


