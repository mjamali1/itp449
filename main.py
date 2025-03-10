import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import math 

def main():

    # gathering and wrangling data
    file_path = 'cumulative_2023.11.07_13.44.30.csv'
    df_nasa = pd.read_csv(file_path, comment='#')
    df_nasa = df_nasa.drop(columns = ['koi_disposition', 'koi_period_err1', 'koi_period_err2', 'koi_eccen_err1', 'koi_eccen_err2',
    'koi_duration_err1', 'koi_duration_err2', 'koi_prad_err1', 'koi_prad_err2', 'koi_sma_err1', 'koi_sma_err2', 'koi_incl_err1',
    'koi_incl_err2', 'koi_teq_err1', 'koi_teq_err2', 'koi_dor_err1', 'koi_dor_err2', 'koi_steff_err1', 'koi_steff_err2',
    'koi_srad_err1', 'koi_srad_err2', 'koi_smass_err1', 'koi_smass_err2'])
    df_nasa = df_nasa.dropna()

    # splitting data into train and test
    X = df_nasa.drop(columns=['koi_pdisposition'])
    y = df_nasa['koi_pdisposition']
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.1, random_state=400, stratify=y)

    # hyperparams
    knn_params = {
        'model__n_neighbors': (1, int(1.5 * np.sqrt(len(X_train))) + 1)
    }

    dt_params = {
        'model__criterion': ['entropy', 'gini'],
        'model__max_depth': np.arange(3,16),
        'model__min_samples_leaf': np.arange(1,11)
    }

    svc_params = {
        'model__kernel': ['rbf'],
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': [0.1, 1, 10]
    }

    models_params = {

        'logistic_regression': (LogisticRegression(), {}),
        'knn': (KNeighborsClassifier(), knn_params),
        'decision_tree': (DecisionTreeClassifier(), dt_params),
        'svc':(SVC(), svc_params)
    }

    # random search without PCA
    non_pca_score = 0
    for name, (model, params) in models_params.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)

        ])
        
        random_search = RandomizedSearchCV(pipe, param_distributions=params, n_iter=2, cv=2, n_jobs = 1, random_state=21)
        random_search.fit(X_train, y_train)

        if random_search.best_score_ > non_pca_score:
            non_pca_score = random_search.best_score_
            non_pca_model = name
            non_pca_pipe = random_search.best_estimator_


    # print(non_pca_score)
    # print(non_pca_model)
    # print(non_pca_pipe)

    # random search with PCA
    pca = PCA(n_components=len(X_train.columns))
    X_pca = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index)

    pca_score = 0
    for name, (model, params) in models_params.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)

        ])  
        
        random_search = RandomizedSearchCV(pipe, param_distributions=params, n_iter = 2, cv=2, n_jobs = 1, random_state=21)
        random_search.fit(X_pca, y_train)

        if random_search.best_score_ > pca_score:
            pca_score = random_search.best_score_
            pca_model = name
            pca_pipe = random_search.best_estimator_

    # print(pca_score)
    # print(pca_model)
    # print(pca_pipe)

    # compare pca and non-pca models
    if pca_score > non_pca_score:
        grid_pipe = pca_pipe
        X_grid = X_pca
        grid_params = models_params[pca_model][1]
        X_test_grid = pd.DataFrame(pca.transform(X_test))
        # print("pca chosen")
    else:
        grid_pipe = non_pca_pipe
        X_grid = X_train
        grid_params = models_params[non_pca_model][1]
        X_test_grid = X_test
        # print("non pca chosen")

    # train final optimized model
    grid_search = GridSearchCV(grid_pipe, param_grid=grid_params, cv=2)
    grid_search.fit(X_grid, y_train)
    estimator = grid_search.best_estimator_

    # final analysis for confusion matrix / classification report
    y_pred = grid_search.predict(X_test_grid)

    # print(sorted(y_test.unique()))
    fig, ax = plt.subplots(figsize=(12,10))
    matrix = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=sorted(y_test.unique()))
    cm_disp.plot(ax = ax)
    ax.set(title = 'Confusion Matrix', xlabel = 'Predicted', ylabel = 'True')
    plt.savefig('exop_conf_matrix.png')

    print('\nclassification report\n')
    print(classification_report(y_test, y_pred))

    # finding most significant attribute
    perm_imp = permutation_importance(estimator, X_test, y_test, n_repeats=5, n_jobs=1, random_state=21)
    result_df = pd.DataFrame({
            'Feature': X.columns,
            'Mean Importance': perm_imp.importances_mean}).sort_values(by='Mean Importance', ascending=False)
    print('\npermutation importance\n')
    print(result_df)

    '''
    Reflection Questions

    Did PCA improve upon or not improve upon the results from when you did not use it?
        - PCA did not improve the results.

    Why might this be?
        - PCA usually improves results when dimensionality (collinearity or multicollinearity) is an issue in the dataset.
        If PCA did not improve the results it suggests that the data does not have any significantly correlated features
        that overpower the results when testing the dataset. It suggests that the number of features is in control as well because 
        the curse of dimensionality -- exponentially growing data -- is not present. 

    Was your model able to classify objects equally across labels? How can you tell?
        - The model was not able to classify objects equally across labels.
        - To come to this conclusion, I analyzed the confusion matrix and the classification report. The confusion matrix visually 
        shows a greater number of misclassifications for the 'FALSE POSITIVE' than the 'CANDIDATE'. Additionally, there is higher
        recall when it is identifying 'CANDIDATE' samples and higher percision when identifying 'FALSE POSITIVE' samples.

    Based on your results, which attribute most significantly influences whether or not an object is an exoplanet? 
    (Hint: you have a way to quantitatively analyze this; it was covered in class.)
        - koi.prad_, the planetary radius is the most significant attribute.

    Describe that attribute in your own words and why you believe it might be most influential. 
    (This is an opinion question so the only way to get it wrong is to not actually reflect.)
        - koi_prad_: the half width of the exoplanet, is the most significant attribute. Using the definition of an exoplanet, a planet 
        that orbits a star outside the solar system, it would make sense that the radius of said object would determine whether or not 
        it is an exoplanet. To orbit a star, an object would have to be the correct size to be a part of its gravitational pull, not too 
        large or too small. If we look at the other features in the dataset, they can be more random and by chance. Radius is more 
        accurately and consistently able to show us if an object is an exoplanet.     
    '''


if __name__ == '__main__':
    main()