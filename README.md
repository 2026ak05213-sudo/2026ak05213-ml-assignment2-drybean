# Dry Bean Classification - ML Assignment 2

## a. Problem Statement
Predict the variety of a dry bean grain from geometric measurements extracted from images of the grain. This is a multi-class classification problem with 7 bean varieties. Automating this step supports seed sorting and quality control, where manual grading is slow and inconsistent. Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on the same data, compared on six metrics and served through a Streamlit app.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository - Dry Bean Dataset (Koklu & Ozkan, 2020), https://archive.ics.uci.edu/dataset/602/dry+bean
- **Instances:** 13,611
- **Features:** 16 (12 dimension features + 4 shape-form features), all numeric
- **Target column:** `Class` with 7 classes. Counts: Dermason 3546, Sira 2636, Seker 2027, Horoz 1928, Cali 1630, Barbunya 1322, Bombay 522. The classes are imbalanced, with Bombay the smallest.
- **Feature examples:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1-4
- **Preprocessing:** no missing values; 68 duplicate rows kept; target label-encoded; StandardScaler placed inside each model pipeline (fitted on training data only)
- **Train/test split:** 80/20 stratified split, `random_state=42` (2,723 test rows saved as `test_data.csv`)

## c. GitHub Repository
<https://github.com/2026ak05213-sudo/2026ak05213-ml-assignment2-drybean>

Live app: <https://2026ak05213-ml-assignment2-drybean-yalvhpfejzgbc7h75ef5n6.streamlit.app/>

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.9041 | 0.9794 | 0.9038 | 0.9041 | 0.9038 | 0.8841 |
| KNN | 0.9137 | 0.9837 | 0.9144 | 0.9137 | 0.9139 | 0.8956 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

Precision, recall and F1 are weighted averages; AUC is one-vs-rest (weighted). Values come from `model/metrics_table.csv`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.9214 and AUC 0.9934, second only to SVM. After scaling, the classes are largely separable by linear boundaries, so a simple model performs strongly. |
| Decision Tree | Accuracy 0.9041, below all models except Naive Bayes. Axis-aligned splits with limited depth (8) struggle with the overlapping, correlated size features, and the lowest AUC (0.9794) reflects coarse probability estimates from leaf nodes. |
| KNN | Accuracy 0.9137 with k=7. Works well on scaled features, but classes with similar shapes overlap in feature space, and prediction is slower than the parametric models. |
| Naive Bayes | Lowest accuracy (0.8979) and MCC (0.8773), yet a high AUC (0.9902). Several features (Area, Perimeter, ConvexArea, EquivDiameter) are strongly correlated, which breaks the independence assumption and hurts hard predictions, while the ranking of class probabilities stays good. |
| Support Vector Machine | Best on every metric: accuracy 0.9221, AUC 0.9937, MCC 0.9059. The RBF kernel captures non-linear class boundaries and copes well with correlated features. Training is slower than the other models. |
| **Overall winner for this dataset** | **Support Vector Machine.** It has the highest accuracy, F1, AUC and MCC, though Logistic Regression is within about 0.1 percentage points and is much cheaper to train. MCC is weighed alongside accuracy because the classes are imbalanced. |

## Repository structure
```
app.py
metrics_utils.py
ML_Assignment2_DryBean.ipynb
requirements.txt
README.md
test_data.csv
model/
```

## How to run locally
```
pip install -r requirements.txt
streamlit run app.py
```
To retrain the models, run `ML_Assignment2_DryBean.ipynb` from top to bottom.
