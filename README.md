# PyTorch_learning
Learning PyTorch and related Data science projects


A. Preprocessing:
    1.  a. from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x_train_std = sc.fit_transform(x_train)
        b. from sklearn.preprocessing import MinMaxScaler
            mms = MinMaxScaler()
            x_train_norm = sc.fit_transform(x_train)
    2a. from sklearn.impute import SimpleImputer
        imr = SimpleImputer(missing_values = np.nan, strategy = 'mean/median/most_frequent/constant', fill_value = 0, copy = True)
        x_train_std = imr.fit_transform(x_train)
    2b. from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors = 3, weights = 'uniform/distance', metric = 'nan_euclidean', copy = False)
        imputer.fit_transform(x_train)
    3. Encoding class labels
        3a. pd.DataSeries.map(dict or lambda func)
        3b. from sklearning preprocessing import LabelEncoder
            color_le = LabelEncoder()
            y = color_le.fit_transform(y_train)
            color_le.inverse_transform(y)
        3c. from sklearn.preprocessing import OneHotEncoder
            color_ohe = OneHotEncoder(categories = 'auto', drop = 'first')
            color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
            or
        3c. from sklearn.compose import ColumnTransformer
            c_transf = ColumnTransformer([
                ('onehot', OneHotEncoder(), [0]), 
                ('nothing', 'passthrough', [1, 2])
            ])
            c_transf.fit_transform(X).astype(np.float32)
        3d. pd.get_dummies(df[['price', 'color', 'size]], drop_first = True)    

B. Classifiers:
    Linear Classifiers:
    1. from sklearn.linear_model import Perception
    ppn = Perceptron(eta0 = 0.1, random_state = 1)

    2. from sklearn.linearmodel import LogisticRegression
    lr = LogisticRegression(C = 1.0, solver = 'lbfgs', multi_class = 'ovr/multinominal')
        C: inverse of L2 regularization strength
    lr = LogisticRegression(penalty = 'l1', C = 1.0, solver = 'liblinear', multi_class = 'ovr/multinominal')
        C: inverse of L1 regularization strength
    lr.intercept_
    lr.coef_

    3. from sklearn.svm import SVC
    svm = SVC(kernel = 'linear', C = 1.0, random_state = 1)

    SGD version of Linear Classifiers:
    from sklearn.linear_model import SGDClassifier
    1. ppn = SGDClassifier(loss = 'perceptron')
    2. lr = SGDClassifier(loss = 'log')
    3. svm = SGDClassifier(loss = 'hinge')

    Nonlinear classifiers: (2., 3. don't care about scaling of X_train)
    1. svm = SVC(kernel = 'rbf', C = 10.0, random_state = 1, gamma = 0.10)
        \kappa(x^(i), x^(j)) = exp(-gamma ||x^(i) - x^(j)||^2)
    2. from sklearn.tree import DecisionTreeClassifier
        tree_model = DecisionTreeClassifier(criterion = 'gini/entropy/log_loss', max_depth = 4, random_state = 1)
    3. from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimator = 25, n_jobs = -1, random_state = 1)
    4. from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric= 'minkowski')

    Classifier-methods:
    ppn.fit(x_train_std, y_train)
    y_pred = ppn.predict(x_test_std)

C. Scores:
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_pred)

    ppn.score(x_test_std, y_test)

D. Feature Selections:
    1. forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1] (descending)

    2. from sklearn.feature_selection import SelectFromModel
        sfm = SelecFromModel(forest, threshold = 0.1, predit = True)
        X_selected = sfm.transform(X_train)

E. Dimension Reduction:
    1. Unsupervised data compression: principal component analysis
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 2)
        X_train_pca = pca.fit_transform(X_train_std)
    2. Supervised data compression linear discriminant analysis
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = 2)
        X_train_lda = lda.fit_transform(X_train_std, y_train)
    3. Nonlinear DR:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components = 2, init = 'pca, random_state = 123)
        X_train_tsne = tsne.fit_transform(X_train)

F. Pipeline, Cross Validation and confusion matrix:
    1. from sklearn.pipeline import make_pipeline
        pipe_lr = make_pipeline(sc, pca, lr)
        pipe_lr.fit(X_train, y_train)
        y_pred = pipe_lr.predict(X_test)
        test_acc = pipe_lr.score(X_test, y_test)
    2. sklearn.model_selection
        a. train -- test splitting
            from sklearn.model_selection import train_test_split <- Simplest splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 1)
        b. K fold cross-validation splitting train set
            from sklearn.model_selection import StratifiedKfold
            kfold = StratifiedKfold(n_splits = 10).split(X_train, y_train)
            scores = []
            for k, (train, test) in enumerate(kfold):
                pipe_lr.fit(X_train[train], y_train[train])
                scores.append(pipe_lr.score(X_train[test], y_train[test]))
 
        
            simpler:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(estimator = pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs = 1)

            mean_acc = np.mean(scores)
            std_acc = np.std(scores)
            print(f'CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
    3. learning_curve
        from sklearn.model_selection import learning_curve
        pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', max_iter=10000))
        train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
    4. validation_curve
        from sklearn.model_selection import validation_curve
        param_range = 10.0 ** (np.arange(-3, 3, 1))
        train_scores, test_scores = validation_curve(estimator=pipe_lr, X = X_train, y = y_train, param_name='logisticregression__C', param_range=param_range.tolist(), cv = 10)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
    5. GridSearchCV
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        
        from sklearn.experimental import enable_halving_search_cv
        from sklearn.model_selection import HalvingRandomSearchCV

        param_grid = [{'svc__C':param_range, 'svc__kernel':['linear']}, {'svc__C':param_range, 'svc__gamma':param_range, 'svc__kernel':['rbf']}]

        gs = RandomizedSearchCV(estimator=pipe_svc, param_distributions=param_grid, scoring='accuracy', cv = 10, refit = True, n_jobs = -1)

    6. More on accuracy
        We have exhausted 'accuracy' as the scoring parameter, what other performance evaluation metrics can we use?
        Confusion matrix: TP + FN + FP + TN = TOTAL
        Accuracy (ACC): (TP + TN) / TOTAL
        Error (ERR): (FP + FN) / TOTAL
        Recall (REC)/TP Rate : TP / (TP + FN) = TP / P
        False Positive Rate : FP / (TN + FP) = FP / N
        Precision (PRE) : TP / (TP + FP)
        F1 Score : 2PRE * REC / (PRE + REC)
        Matthews Correlation coefficient MCC : (TP * TN - FP * FN) / sqrt{(TP + FP)(TN + FP)(TP + FN)(TN + FN)}
        Receiver Operating Characteristic (ROC) graphs

        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

        from sklearn.metrics import make_scorer
        scorer = make_scorer(f1_score, pos_label = 0)

    7. Imbalanced Dataset
        Class Imbalances:

        X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
        y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

        If 90% of the dataset belong to one category, the precision score is not necessarily the optimal criterion for learning. Moreover, this imbalance leads to suboptimal learning progression.
        Solution #1: activate class_weighted = 'balanced' (or assign a predefined dictionary {0:0.9, 1:0.1, ... })
        Solution #2: Change the sampling with sklearn.utils.resample
            from sklearn.utils import resample
            X_upsampled, y_upsampled = resample(X_imb[y_imb == 1], y_imb[y_imb == 1], replace=True, n_samples=X_imb[y_imb == 0].shape[0], random_state=123)
