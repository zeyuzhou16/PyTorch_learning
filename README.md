# PyTorch_learning

Learning PyTorch and related Data Science projects

---

## Table of Contents

- [A. Preprocessing](#a-preprocessing)
- [B. Classifiers](#b-classifiers)
- [C. Scores](#c-scores)
- [D. Feature Selection](#d-feature-selection)
- [E. Dimension Reduction](#e-dimension-reduction)
- [F. Pipeline, Cross Validation, and Confusion Matrix](#f-pipeline-cross-validation-and-confusion-matrix)
- [G. Imbalanced Dataset](#g-imbalanced-dataset)

---

## A. Preprocessing

### 1. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard scaling
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)

# Min-max scaling
mms = MinMaxScaler()
x_train_norm = mms.fit_transform(x_train)
```

### 2. Imputation (Handling Missing Values)

```python
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

# Simple imputer
imr = SimpleImputer(missing_values=np.nan, strategy='mean')  # or 'median', 'most_frequent', 'constant'
x_train_imp = imr.fit_transform(x_train)

# KNN imputer
imputer = KNNImputer(n_neighbors=3)
x_train_knn = imputer.fit_transform(x_train)
```

### 3. Encoding Class Labels

- **Using pandas map:**
    ```python
    df['class'] = df['class'].map({'A': 0, 'B': 1})
    ```
- **LabelEncoder:**
    ```python
    from sklearn.preprocessing import LabelEncoder

    color_le = LabelEncoder()
    y = color_le.fit_transform(y_train)
    color_le.inverse_transform(y)
    ```
- **OneHotEncoder:**
    ```python
    from sklearn.preprocessing import OneHotEncoder

    color_ohe = OneHotEncoder(drop='first')
    encoded = color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
    ```
- **ColumnTransformer:**
    ```python
    from sklearn.compose import ColumnTransformer

    c_transf = ColumnTransformer([
        ('onehot', OneHotEncoder(), [0]), 
        ('nothing', 'passthrough', [1, 2])
    ])
    transformed = c_transf.fit_transform(X).astype(np.float32)
    ```
- **pandas get_dummies:**
    ```python
    pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)
    ```

---

## B. Classifiers

### 1. Linear Classifiers

- **Perceptron**
    ```python
    from sklearn.linear_model import Perceptron

    ppn = Perceptron(eta0=0.1, random_state=1)
    ```
- **Logistic Regression**
    ```python
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')  # C: inverse of L2 regularization strength
    lr_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')  # L1 regularization
    lr.intercept_
    lr.coef_
    ```
- **Support Vector Machine (Linear)**
    ```python
    from sklearn.svm import SVC

    svm = SVC(kernel='linear', C=1.0, random_state=1)
    ```

### 2. SGD Version of Linear Classifiers

```python
from sklearn.linear_model import SGDClassifier

ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')
```

### 3. Nonlinear Classifiers

- **SVM (RBF Kernel):**
    ```python
    svm = SVC(kernel='rbf', C=10.0, random_state=1, gamma=0.10)
    # RBF: kappa(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
    ```
- **Decision Tree:**
    ```python
    from sklearn.tree import DecisionTreeClassifier

    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    # criterion can be 'gini', 'entropy', or 'log_loss'
    ```
- **Random Forest:**
    ```python
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=25, n_jobs=-1, random_state=1)
    ```
- **K-Nearest Neighbors:**
    ```python
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    ```

### 4. Classifier Methods

```python
ppn.fit(x_train_std, y_train)
y_pred = ppn.predict(x_test_std)
```

---

## C. Scores

### Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
ppn.score(x_test_std, y_test)
```

---

## D. Feature Selection

```python
# Feature importances from Random Forest
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]  # Descending

# SelectFromModel for feature selection
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
```

---

## E. Dimension Reduction

### 1. Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
```

### 2. Linear Discriminant Analysis (LDA)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
```

### 3. Nonlinear Dimension Reduction (t-SNE)

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, init='pca', random_state=123)
X_train_tsne = tsne.fit_transform(X_train)
```

---

## F. Pipeline, Cross Validation, and Confusion Matrix

### 1. Pipeline

```python
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(sc, pca, lr)
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
```

### 2. Model Selection

#### a. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=1
)
```

#### b. K-Fold Cross Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    scores.append(pipe_lr.score(X_train[test], y_train[test]))

# Simpler:
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)

mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f'CV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')
```

### 3. Learning Curve

```python
from sklearn.model_selection import learning_curve

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', max_iter=10000))
train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1
)
```

### 4. Validation Curve

```python
from sklearn.model_selection import validation_curve

param_range = 10.0 ** np.arange(-3, 3, 1)
train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name='logisticregression__C',
    param_range=param_range.tolist(),
    cv=10
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
```

### 5. Grid Search & Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

param_grid = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}
]

gs = RandomizedSearchCV(
    estimator=pipe_lr, param_distributions=param_grid,
    scoring='accuracy', cv=10, refit=True, n_jobs=-1
)
```

### 6. Performance Metrics

Confusion matrix and other metrics:
- **Confusion matrix:** TP + FN + FP + TN = TOTAL
- **Accuracy (ACC):** (TP + TN) / TOTAL
- **Error (ERR):** (FP + FN) / TOTAL
- **Recall (REC)/TP Rate:** TP / (TP + FN) = TP / P
- **False Positive Rate:** FP / (TN + FP) = FP / N
- **Precision (PRE):** TP / (TP + FP)
- **F1 Score:** 2 * PRE * REC / (PRE + REC)
- **Matthews Correlation Coefficient (MCC):** (TP * TN - FP * FN) / sqrt{(TP + FP)(TN + FP)(TP + FN)(TN + FN)}

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, make_scorer

scorer = make_scorer(f1_score, pos_label=0)
```

---

## G. Imbalanced Dataset

When the dataset is imbalanced (e.g., 90% of data in one class), accuracy is not a sufficient metric. Consider:

1. **Class weights:**  
    Use `class_weight='balanced'` or a custom dictionary in model parameters.

2. **Resampling:**  
    ```python
    from sklearn.utils import resample

    X_upsampled, y_upsampled = resample(
        X_imb[y_imb == 1], y_imb[y_imb == 1],
        replace=True, n_samples=X_imb[y_imb == 0].shape[0], random_state=123
    )
    ```

---

*End of README*
