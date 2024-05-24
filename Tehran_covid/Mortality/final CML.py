from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_excel("3.xlsx")


df = df.dropna(
    subset=[
        "Outcome_InhospitalMortality",
        "TM_S_Intubation",
        "Outcome_ICUadmission",
        "TM_S_Dialysis",
    ]
)


num_cols = df.shape[1]
min_non_null = int(0.3 * num_cols)
df = df.dropna(thresh=min_non_null)


df_internal_validation = df[df["Patient_Hospital"] != "Hospital2"]
df_external_validation = df[df["Patient_Hospital"] == "Hospital2"]
df_internal_validation = df_internal_validation.drop(
    columns=["Patient_Hospital"])
df_external_validation = df_external_validation.drop(
    columns=["Patient_Hospital"])


X = df_internal_validation.drop(
    columns=[
        "Outcome_InhospitalMortality",
        "TM_S_Intubation",
        "Outcome_ICUadmission",
        "TM_S_Dialysis",
    ]
)
y = df_internal_validation[
    [
        "Outcome_InhospitalMortality",
        "TM_S_Intubation",
        "Outcome_ICUadmission",
        "TM_S_Dialysis",
    ]
]


imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = imputer.fit_transform(X)

y_ex = pd.DataFrame(y.loc[:, "Outcome_InhospitalMortality"])


y_train = y[["Outcome_InhospitalMortality"]]


X_ex = df_external_validation.drop(
    columns=[
        "Outcome_InhospitalMortality",
        "TM_S_Intubation",
        "Outcome_ICUadmission",
        "TM_S_Dialysis",
    ]
)
y_ex = df_external_validation[
    [
        "Outcome_InhospitalMortality",
        "TM_S_Intubation",
        "Outcome_ICUadmission",
        "TM_S_Dialysis",
    ]
]


imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed_ex = imputer.fit_transform(X_ex)

y_ex = pd.DataFrame(y_ex.loc[:, "Outcome_InhospitalMortality"])


X_train, X_test, y_train, y_test = train_test_split(
    df_imputed, y, test_size=0.3, random_state=42
)


y_train = y_train["Outcome_InhospitalMortality"]
y_test = y_test["Outcome_InhospitalMortality"]


arr_ex = df_imputed_ex  # .to_numpy()

transform = preprocessing.StandardScaler()
df_ex_normalize = transform.fit_transform(arr_ex)
df_ex_normalize = pd.DataFrame(df_ex_normalize)
df_ex_normalize.columns = X_ex.columns.tolist()

X_ex = df_ex_normalize


arr_tain = X_train  # .to_numpy()
transform = preprocessing.StandardScaler()
df_normalize = transform.fit_transform(arr_tain)
df_normalize = pd.DataFrame(df_normalize)
df_normalize.columns = X.columns.tolist()
X_train = df_normalize


arr = X_test  # .to_numpy()
transform = preprocessing.StandardScaler()
df_test_normalize = transform.fit_transform(arr)
df_test_normalize = pd.DataFrame(df_test_normalize)
df_test_normalize.columns = X.columns.tolist()
X_test = df_test_normalize


def lasso_MO_ICU(X_train, y_train, X_test, X_ex):

    lasso = Lasso(alpha=0.0001, random_state=42)
    lasso.fit(X_train, y_train)

    absolute_coeffs = np.abs(lasso.coef_)
    sorted_indices = np.argsort(absolute_coeffs)[::-1]

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_ex = pd.DataFrame(X_ex)

    selected_feature_indices = sorted_indices[:40]
    selected_feature_indices = np.where(lasso.coef_ != 0)[0]

    X_train_selected_Mortality_ICU = X_train.iloc[:, selected_feature_indices]
    X_test_selected_Mortality_ICU = X_test.iloc[:, selected_feature_indices]
    X_ex = X_ex.iloc[:, selected_feature_indices]
    return X_train_selected_Mortality_ICU, X_test_selected_Mortality_ICU, X_ex


X_train, X_test, X_ex = lasso_MO_ICU(X_train, y_train, X_test, X_ex)


X_train = X_train.drop(columns=["L1_BloodGroup_First"])
X_test = X_test.drop(columns=["L1_BloodGroup_First"])
X_ex = X_ex.drop(columns=["L1_BloodGroup_First"])


def balancing_dataset(X_train, y_train):

    tl = RandomUnderSampler(random_state=42)
    X_resampled_train, y_resampled_train = tl.fit_resample(X_train, y_train)
    X_resampled_test, y_resampled_test = tl.fit_resample(X_test, y_test)

    return (X_resampled_train, y_resampled_train, X_resampled_test, y_resampled_test)


a, b, X_test, y_test = balancing_dataset(X_train, y_train)


X_train, y_train = a, b


X_train = np.array(X_train)
X_test = np.array(X_test)


def logistic_regression_classifier(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_lr = {"random_state": [42]}

    lr = LogisticRegression(random_state=42)

    grid_search_lr = GridSearchCV(
        estimator=lr, param_grid=parameters_lr, cv=5, n_jobs=-1
    )

    logreg_cv = grid_search_lr.fit(X_train, y_train)

    best_classifier_lr = logreg_cv.best_estimator_

    y_predicted_Mortality_lr = best_classifier_lr.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted_Mortality_lr)
    cm = confusion_matrix(y_test, y_predicted_Mortality_lr)
    precision = precision_score(y_test, y_predicted_Mortality_lr)
    recall = recall_score(y_test, y_predicted_Mortality_lr)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_predicted_Mortality_lr)

    Logistic_Regression = {}
    Logistic_Regression["accuracy"] = accuracy
    Logistic_Regression["precision"] = precision
    Logistic_Regression["recall"] = recall
    Logistic_Regression["specificity"] = specificity
    Logistic_Regression["f1"] = f1
    Logistic_Regression["AUC"] = roc_auc_score(
        y_test, y_predicted_Mortality_lr)

    y_plot = {}
    y_plot["y_true"] = y_test
    y_plot["y_predicted"] = y_predicted_Mortality_lr

    y_predicted_Mortality_boost = best_classifier_lr.predict(X_ex)
    accuracy = accuracy_score(y_ex, y_predicted_Mortality_boost)
    cm = confusion_matrix(y_ex, y_predicted_Mortality_boost)
    precision = precision_score(y_ex, y_predicted_Mortality_boost)
    recall = recall_score(y_ex, y_predicted_Mortality_boost)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_ex, y_predicted_Mortality_boost)

    Logistic_Regression["accuracy_ex"] = accuracy
    Logistic_Regression["precision_ex"] = precision
    Logistic_Regression["recall_ex"] = recall
    Logistic_Regression["specificity_ex"] = specificity
    Logistic_Regression["f1_ex"] = f1
    Logistic_Regression["AUC_ex"] = roc_auc_score(
        y_ex, y_predicted_Mortality_boost)

    y_plot["y_true_ex"] = y_ex
    y_plot["y_predicted_ex"] = y_predicted_Mortality_boost

    Logistic_Regression = (Logistic_Regression, y_plot)

    return Logistic_Regression


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, X_ex, y_ex):

    svm_classifier = svm.SVC(random_state=42)

    parameters_svm = {"random_state": [42]}

    grid_search_svm = GridSearchCV(
        estimator=svm_classifier, param_grid=parameters_svm, cv=5
    )

    svm_cv = grid_search_svm.fit(X_train, y_train)

    best_classifier_svm = svm_cv.best_estimator_

    y_predicted_Mortality_svm = best_classifier_svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted_Mortality_svm)
    cm = confusion_matrix(y_test, y_predicted_Mortality_svm)
    precision = precision_score(y_test, y_predicted_Mortality_svm)
    recall = recall_score(y_test, y_predicted_Mortality_svm)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_predicted_Mortality_svm)

    svm_results = {}
    svm_results["accuracy"] = accuracy
    svm_results["precision"] = precision
    svm_results["recall"] = recall
    svm_results["specificity"] = specificity
    svm_results["f1"] = f1
    svm_results["AUC"] = roc_auc_score(y_test, y_predicted_Mortality_svm)

    y_plot = {}
    y_plot["y_true"] = y_test
    y_plot["y_predicted"] = y_predicted_Mortality_svm

    y_predicted_Mortality_boost = best_classifier_svm.predict(X_ex)
    accuracy = accuracy_score(y_ex, y_predicted_Mortality_boost)
    cm = confusion_matrix(y_ex, y_predicted_Mortality_boost)
    precision = precision_score(y_ex, y_predicted_Mortality_boost)
    recall = recall_score(y_ex, y_predicted_Mortality_boost)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_ex, y_predicted_Mortality_boost)

    svm_results["accuracy_ex"] = accuracy
    svm_results["precision_ex"] = precision
    svm_results["recall_ex"] = recall
    svm_results["specificity_ex"] = specificity
    svm_results["f1_ex"] = f1
    svm_results["AUC_ex"] = roc_auc_score(y_ex, y_predicted_Mortality_boost)

    y_plot["y_true_ex"] = y_ex
    y_plot["y_predicted_ex"] = y_predicted_Mortality_boost

    svm_results = (svm_results, y_plot)

    return svm_results


def train_and_evaluate_tree(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_tree = {"random_state": [42]}

    tree = DecisionTreeClassifier(random_state=42)

    grid_search_tree = GridSearchCV(
        estimator=tree, param_grid=parameters_tree, cv=5)

    tree_cv = grid_search_tree.fit(X_train, y_train)

    best_classifier_tree = tree_cv.best_estimator_

    y_predicted_Mortality_tree = best_classifier_tree.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted_Mortality_tree)
    cm = confusion_matrix(y_test, y_predicted_Mortality_tree)
    precision = precision_score(y_test, y_predicted_Mortality_tree)
    recall = recall_score(y_test, y_predicted_Mortality_tree)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_predicted_Mortality_tree)

    tree_results = {}
    tree_results["accuracy"] = accuracy
    tree_results["precision"] = precision
    tree_results["recall"] = recall
    tree_results["specificity"] = specificity
    tree_results["f1"] = f1
    tree_results["AUC"] = roc_auc_score(y_test, y_predicted_Mortality_tree)

    y_plot = {}
    y_plot["y_true"] = y_test
    y_plot["y_predicted"] = y_predicted_Mortality_tree

    y_predicted_Mortality_boost = best_classifier_tree.predict(X_ex)
    accuracy = accuracy_score(y_ex, y_predicted_Mortality_boost)
    cm = confusion_matrix(y_ex, y_predicted_Mortality_boost)
    precision = precision_score(y_ex, y_predicted_Mortality_boost)
    recall = recall_score(y_ex, y_predicted_Mortality_boost)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_ex, y_predicted_Mortality_boost)

    tree_results["accuracy_ex"] = accuracy
    tree_results["precision_ex"] = precision
    tree_results["recall_ex"] = recall
    tree_results["specificity_ex"] = specificity
    tree_results["f1_ex"] = f1
    tree_results["AUC_ex"] = roc_auc_score(y_ex, y_predicted_Mortality_boost)

    y_plot["y_true_ex"] = y_ex
    y_plot["y_predicted_ex"] = y_predicted_Mortality_boost

    tree_results = (tree_results, y_plot)

    return tree_results


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_knn = {}

    knn = KNeighborsClassifier()

    grid_search_knn = GridSearchCV(
        estimator=knn, param_grid=parameters_knn, cv=5)

    knn_cv = grid_search_knn.fit(X_train, y_train)

    best_classifier_knn = knn_cv.best_estimator_

    y_predicted_Mortality_knn = best_classifier_knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted_Mortality_knn)
    cm = confusion_matrix(y_test, y_predicted_Mortality_knn)
    precision = precision_score(y_test, y_predicted_Mortality_knn)
    recall = recall_score(y_test, y_predicted_Mortality_knn)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_predicted_Mortality_knn)

    knn_results = {}
    knn_results["accuracy"] = accuracy
    knn_results["precision"] = precision
    knn_results["recall"] = recall
    knn_results["specificity"] = specificity
    knn_results["f1"] = f1
    knn_results["AUC"] = roc_auc_score(y_test, y_predicted_Mortality_knn)

    y_plot = {}
    y_plot["y_true"] = y_test
    y_plot["y_predicted"] = y_predicted_Mortality_knn

    y_predicted_Mortality_boost = best_classifier_knn.predict(X_ex)
    accuracy = accuracy_score(y_ex, y_predicted_Mortality_boost)
    cm = confusion_matrix(y_ex, y_predicted_Mortality_boost)
    precision = precision_score(y_ex, y_predicted_Mortality_boost)
    recall = recall_score(y_ex, y_predicted_Mortality_boost)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_ex, y_predicted_Mortality_boost)

    knn_results["accuracy_ex"] = accuracy
    knn_results["precision_ex"] = precision
    knn_results["recall_ex"] = recall
    knn_results["specificity_ex"] = specificity
    knn_results["f1_ex"] = f1
    knn_results["AUC_ex"] = roc_auc_score(y_ex, y_predicted_Mortality_boost)

    y_plot["y_true_ex"] = y_ex
    y_plot["y_predicted_ex"] = y_predicted_Mortality_boost

    knn_results = (knn_results, y_plot)

    return knn_results


def train_and_evaluate_forest(X_train, y_train, X_test, y_test, X_ex, y_ex):
    parameters_forest = {"random_state": [42]}

    forest = RandomForestClassifier(random_state=42)

    grid_search_forest = GridSearchCV(
        estimator=forest, param_grid=parameters_forest, cv=5
    )

    forest_cv = grid_search_forest.fit(X_train, y_train)

    best_classifier_forest = forest_cv.best_estimator_

    y_predicted_Mortality_forest = best_classifier_forest.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted_Mortality_forest)
    cm = confusion_matrix(y_test, y_predicted_Mortality_forest)
    precision = precision_score(y_test, y_predicted_Mortality_forest)
    recall = recall_score(y_test, y_predicted_Mortality_forest)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_predicted_Mortality_forest)

    forest_results = {}
    forest_results["accuracy"] = accuracy
    forest_results["precision"] = precision
    forest_results["recall"] = recall
    forest_results["specificity"] = specificity
    forest_results["f1"] = f1
    forest_results["AUC"] = roc_auc_score(y_test, y_predicted_Mortality_forest)

    y_plot = {}
    y_plot["y_true"] = y_test
    y_plot["y_predicted"] = y_predicted_Mortality_forest

    y_predicted_Mortality_boost = best_classifier_forest.predict(X_ex)
    accuracy = accuracy_score(y_ex, y_predicted_Mortality_boost)
    cm = confusion_matrix(y_ex, y_predicted_Mortality_boost)
    precision = precision_score(y_ex, y_predicted_Mortality_boost)
    recall = recall_score(y_ex, y_predicted_Mortality_boost)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_ex, y_predicted_Mortality_boost)

    forest_results["accuracy_ex"] = accuracy
    forest_results["precision_ex"] = precision
    forest_results["recall_ex"] = recall
    forest_results["specificity_ex"] = specificity
    forest_results["f1_ex"] = f1
    forest_results["AUC_ex"] = roc_auc_score(y_ex, y_predicted_Mortality_boost)

    y_plot["y_true_ex"] = y_ex
    y_plot["y_predicted_ex"] = y_predicted_Mortality_boost

    return forest_results, y_plot


def train_and_evaluate_neural(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_neural = {
        # You can adjust the architecture here
        "hidden_layer_sizes": [(100,), (50, 50)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001],
        "max_iter": [200],
        "random_state": [42],
        "early_stopping": [True],
        "validation_fraction": [0.1],
        "n_iter_no_change": [10],
    }

    neural = MLPClassifier(random_state=42)

    grid_search_neural = GridSearchCV(
        estimator=neural, param_grid=parameters_neural, cv=5
    )

    neural_cv = grid_search_neural.fit(X_train, y_train)

    best_classifier_neural = neural_cv.best_estimator_

    y_predicted_Mortality_neural = best_classifier_neural.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted_Mortality_neural)
    cm = confusion_matrix(y_test, y_predicted_Mortality_neural)
    precision = precision_score(y_test, y_predicted_Mortality_neural)
    recall = recall_score(y_test, y_predicted_Mortality_neural)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_predicted_Mortality_neural)

    neural_results = {}
    neural_results["accuracy"] = accuracy
    neural_results["precision"] = precision
    neural_results["recall"] = recall
    neural_results["specificity"] = specificity
    neural_results["f1"] = f1
    neural_results["AUC"] = roc_auc_score(y_test, y_predicted_Mortality_neural)

    y_plot = {}
    y_plot["y_true"] = y_test
    y_plot["y_predicted"] = y_predicted_Mortality_neural

    y_predicted_Mortality_boost = best_classifier_neural.predict(X_ex)
    accuracy = accuracy_score(y_ex, y_predicted_Mortality_boost)
    cm = confusion_matrix(y_ex, y_predicted_Mortality_boost)
    precision = precision_score(y_ex, y_predicted_Mortality_boost)
    recall = recall_score(y_ex, y_predicted_Mortality_boost)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_ex, y_predicted_Mortality_boost)

    neural_results["accuracy_ex"] = accuracy
    neural_results["precision_ex"] = precision
    neural_results["recall_ex"] = recall
    neural_results["specificity_ex"] = specificity
    neural_results["f1_ex"] = f1
    neural_results["AUC_ex"] = roc_auc_score(y_ex, y_predicted_Mortality_boost)

    y_plot["y_true_ex"] = y_ex
    y_plot["y_predicted_ex"] = y_predicted_Mortality_boost

    return neural_results, y_plot


def train_and_evaluate_boost(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_boost = {"random_state": [42]}

    boost = XGBClassifier(random_state=42)

    grid_search_boost = GridSearchCV(
        estimator=boost, param_grid=parameters_boost, cv=5)

    boost_cv = grid_search_boost.fit(X_train, y_train)

    best_classifier_boost = boost_cv.best_estimator_

    y_predicted_Mortality_boost = best_classifier_boost.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted_Mortality_boost)
    cm = confusion_matrix(y_test, y_predicted_Mortality_boost)
    precision = precision_score(y_test, y_predicted_Mortality_boost)
    recall = recall_score(y_test, y_predicted_Mortality_boost)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_predicted_Mortality_boost)

    boost_results = {}
    boost_results["accuracy"] = accuracy
    boost_results["precision"] = precision
    boost_results["recall"] = recall
    boost_results["specificity"] = specificity
    boost_results["f1"] = f1
    boost_results["AUC"] = roc_auc_score(y_test, y_predicted_Mortality_boost)
    y_plot = {}
    y_plot["y_true"] = y_test
    y_plot["y_predicted"] = y_predicted_Mortality_boost

    y_predicted_Mortality_boost = best_classifier_boost.predict(X_ex)
    accuracy = accuracy_score(y_ex, y_predicted_Mortality_boost)
    cm = confusion_matrix(y_ex, y_predicted_Mortality_boost)
    precision = precision_score(y_ex, y_predicted_Mortality_boost)
    recall = recall_score(y_ex, y_predicted_Mortality_boost)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_ex, y_predicted_Mortality_boost)

    boost_results["accuracy_ex"] = accuracy
    boost_results["precision_ex"] = precision
    boost_results["recall_ex"] = recall
    boost_results["specificity_ex"] = specificity
    boost_results["f1_ex"] = f1
    boost_results["AUC_ex"] = roc_auc_score(y_ex, y_predicted_Mortality_boost)
    y_plot["y_true_ex"] = y_ex
    y_plot["y_predicted_ex"] = y_predicted_Mortality_boost

    return boost_results, y_plot


list_Outcome_InhospitalMortality = []

y_train_ = pd.DataFrame(y_train)
y_test_ = pd.DataFrame(y_test)


y_train_ = np.array(y_train)
y_test_ = np.array(y_test)


X_test = np.array(X_test)
y_ex_ = np.array(y_ex["Outcome_InhospitalMortality"].tolist())
X_ex = np.array(X_ex)


y_ex_


list_Outcome_InhospitalMortality.extend(
    [
        logistic_regression_classifier(
            X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_svm(X_train, y_train_, X_test,
                               y_test_, X_ex, y_ex_),
        train_and_evaluate_tree(
            X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_knn(X_train, y_train_, X_test,
                               y_test_, X_ex, y_ex_),
        train_and_evaluate_forest(
            X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_neural(
            X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_boost(
            X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
    ]
)


result_dic_list_Outcome_InhospitalMortality = dict(
    zip(
        ["logistic_regression", "svm", "tree", "knn", "forest", "neural", "boost"],
        list_Outcome_InhospitalMortality,
    )
)


merged_dict = {}

# List of dictionary names and their corresponding dictionaries
dict_list = [
    ("Outcome_InhospitalMortality", result_dic_list_Outcome_InhospitalMortality)
]

# Merge the dictionaries
for name, result_dict in dict_list:
    merged_dict[name] = result_dict

# The merged_dict now contains all the dictionaries merged together


df = pd.DataFrame(merged_dict)


df = df.transpose()


df.reset_index(inplace=True)
df.rename(columns={"index": "Method"}, inplace=True)


data = []

for outcome, models in merged_dict.items():
    for model, metrics in models.items():
        (
            accuracy,
            precision,
            recall,
            specificity,
            f1,
            AUC,
            accuracy_ex,
            precision_ex,
            recall_ex,
            specificity_ex,
            f1_ex,
            AUC_ex,
        ) = (
            metrics[0]["accuracy"],
            metrics[0]["precision"],
            metrics[0]["recall"],
            metrics[0]["specificity"],
            metrics[0]["f1"],
            metrics[0]["AUC"],
            metrics[0]["accuracy_ex"],
            metrics[0]["precision_ex"],
            metrics[0]["recall_ex"],
            metrics[0]["specificity_ex"],
            metrics[0]["f1_ex"],
            metrics[0]["AUC_ex"],
        )
        y_true, y_predicted, y_true_ex, y_predicted_ex = (
            metrics[1]["y_true"],
            metrics[1]["y_predicted"],
            metrics[1]["y_true_ex"],
            metrics[1]["y_predicted_ex"],
        )
        data.append(
            [
                outcome,
                model,
                accuracy,
                precision,
                recall,
                specificity,
                f1,
                AUC,
                accuracy_ex,
                precision_ex,
                recall_ex,
                specificity_ex,
                f1_ex,
                AUC_ex,
                y_true.tolist(),
                y_predicted.tolist(),
                y_true_ex.tolist(),
                y_predicted_ex.tolist(),
            ]
        )

columns = [
    "Outcome",
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "AUC",
    "Accuracy_ex",
    "Precision_ex",
    "Recall_ex",
    "Specificity_ex",
    "F1_ex",
    "AUC_ex",
    "y_true",
    "y_predicted",
    "y_true_ex",
    "y_predicted_ex",
]

df = pd.DataFrame(data, columns=columns)


# df.to_excel('CML.xlsx',index=False)
