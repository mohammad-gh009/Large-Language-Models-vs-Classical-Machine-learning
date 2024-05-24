from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# # Train and Test sets


df = pd.read_excel("3.xlsx")


# ooutcome columns: 1.are Outcome_InhospitalMortality
df = df.dropna(
    subset=[
        "Outcome_InhospitalMortality",
        "TM_S_Intubation",
        "Outcome_ICUadmission",
        "TM_S_Dialysis",
    ]
)




def clean_data(df):
    # Drop duplicate rows across all columns
    df = df.drop_duplicates()
    # Drop columns: 'L1_BloodGroup_First', 'Symptom_Hemiparesia' and 19 other columns features with more than 20 percent mising values
    df = df.drop(
        columns=[
            "L1_BloodGroup_First",
            "Symptom_Hemiparesia",
            "MH_PregnanAcy",
            "symtpm_to_referral",
            "VS_RR",
            "VS_T",
            "LAB_INR_First",
            "LAB_PT_First",
            "LAB_PTT_First",
            "LAB_CPK_First",
            "LAB_ESR_First",
            "LAB_ALKP_First",
            "LAB_K_First",
            "LAB_NA_First",
            "LAB_CR_1",
            "LAB_MCV_1",
            "LAB_HB_1",
            "LAB_PLT_1",
            "LAB_NEUT_1",
            "LAB_LYMPHH_1",
            "LAB_WBC_1",
        ]
    )
    return df


df = clean_data(df.copy())


# external validation
df_internal_validation = df[df["Patient_Hospital"] != "Hospital2"]
df_external_validation = df[df["Patient_Hospital"] == "Hospital2"]
df_internal_validation = df_internal_validation.drop(columns=["Patient_Hospital"])
df_external_validation = df_external_validation.drop(columns=["Patient_Hospital"])


# train test split
X = df_internal_validation.drop(
    columns=[
        "Outcome_InhospitalMortality",
        "TM_S_Intubation",
        "Outcome_ICUadmission",
        "TM_S_Dialysis",
    ]
)
y = df_internal_validation[["Outcome_InhospitalMortality"]]


imputer = IterativeImputer(max_iter=10, random_state=0)
X_im = imputer.fit_transform(X)
X = pd.DataFrame(X_im, columns=X.columns)


X_ex = df_external_validation.drop(
    columns=[
        "Outcome_InhospitalMortality",
        "TM_S_Intubation",
        "Outcome_ICUadmission",
        "TM_S_Dialysis",
    ]
)
y_ex = df_external_validation[["Outcome_InhospitalMortality"]]


X_ex_im = imputer.fit_transform(X_ex)

X_ex = pd.DataFrame(X_ex_im, columns=X_ex.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


y_train = y_train["Outcome_InhospitalMortality"]
y_test = y_test["Outcome_InhospitalMortality"]


# Assuming df is your DataFrame
columns_to_standardize = [
    "Demographic_Age",
    "VS_Systolic BP",
    "VS_diastolic BP",
    "VS_PR",
    "VS_O2satwithoutsupp",
]

# Create a StandardScaler instance
transform = StandardScaler()

# Fit and transform the selected columns
X_ex[columns_to_standardize] = transform.fit_transform(X_ex[columns_to_standardize])


# Assuming df is your DataFrame
columns_to_standardize = [
    "Demographic_Age",
    "VS_Systolic BP",
    "VS_diastolic BP",
    "VS_PR",
    "VS_O2satwithoutsupp",
]

# Create a StandardScaler instance
transform = StandardScaler()

# Fit and transform the selected columns
X_train[columns_to_standardize] = transform.fit_transform(
    X_train[columns_to_standardize]
)


# Assuming df is your DataFrame
columns_to_standardize = [
    "Demographic_Age",
    "VS_Systolic BP",
    "VS_diastolic BP",
    "VS_PR",
    "VS_O2satwithoutsupp",
]

# Create a StandardScaler instance
transform = StandardScaler()

# Fit and transform the selected columns
X_test[columns_to_standardize] = transform.fit_transform(X_test[columns_to_standardize])


def lasso_MO_ICU(X_train, y_train, X_test, X_ex):
    # Perform feature selection using LASSO on the training set

    lasso = Lasso(alpha=0.0001, random_state=42)
    lasso.fit(X_train, y_train)

    # Get the absolute coefficients and sort them
    absolute_coeffs = np.abs(lasso.coef_)
    sorted_indices = np.argsort(absolute_coeffs)[::-1]

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_ex = pd.DataFrame(X_ex)
    # Select the top 2 features with the highest coefficients
    selected_feature_indices = sorted_indices[:40]
    selected_feature_indices = np.where(lasso.coef_ != 0)[0]

    # Apply feature selection to both the training and test sets
    X_train_selected_Mortality_ICU = X_train.iloc[:, selected_feature_indices]
    X_test_selected_Mortality_ICU = X_test.iloc[:, selected_feature_indices]
    X_ex = X_ex.iloc[:, selected_feature_indices]
    return X_train_selected_Mortality_ICU, X_test_selected_Mortality_ICU, X_ex


X_train, X_test, X_ex = lasso_MO_ICU(X_train, y_train, X_test, X_ex)


def balancing_dataset(X_train, y_train):

    tl = RandomUnderSampler(random_state=42)
    X_resampled_train, y_resampled_train = tl.fit_resample(X_train, y_train)
    X_resampled_test, y_resampled_test = tl.fit_resample(X_test, y_test)

    return (X_resampled_train, y_resampled_train, X_resampled_test, y_resampled_test)


X_train, y_train, X_test, y_test = balancing_dataset(X_train, y_train)


# >>>Now dataset is cleaned and normalized


def evaluate_classifier(classifier, X_test, y_test, X_ex, y_ex):
    y_predicted = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    cm = confusion_matrix(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_predicted)

    results = {}
    results["accuracy"] = accuracy
    results["precision"] = precision
    results["recall"] = recall
    results["specificity"] = specificity
    results["f1"] = f1
    results["AUC"] = roc_auc_score(y_test, y_predicted)

    y_plot = {}
    y_plot["y_true"] = y_test
    y_plot["y_predicted"] = y_predicted

    y_predicted_ex = classifier.predict(X_ex)
    accuracy_ex = accuracy_score(y_ex, y_predicted_ex)
    cm_ex = confusion_matrix(y_ex, y_predicted_ex)
    precision_ex = precision_score(y_ex, y_predicted_ex)
    recall_ex = recall_score(y_ex, y_predicted_ex)
    specificity_ex = cm_ex[0, 0] / (cm_ex[0, 0] + cm_ex[0, 1])
    f1_ex = f1_score(y_ex, y_predicted_ex)

    results["accuracy_ex"] = accuracy_ex
    results["precision_ex"] = precision_ex
    results["recall_ex"] = recall_ex
    results["specificity_ex"] = specificity_ex
    results["f1_ex"] = f1_ex
    results["AUC_ex"] = roc_auc_score(y_ex, y_predicted_ex)

    y_plot["y_true_ex"] = y_ex
    y_plot["y_predicted_ex"] = y_predicted_ex

    return results, y_plot


def logistic_regression_classifier(X_train, y_train, X_test, y_test, X_ex, y_ex):
    parameters_lr = {"random_state": [42]}
    lr = LogisticRegression(random_state=42)
    grid_search_lr = GridSearchCV(
        estimator=lr, param_grid=parameters_lr, cv=5, n_jobs=-1
    )
    logreg_cv = grid_search_lr.fit(X_train, y_train)
    best_classifier_lr = logreg_cv.best_estimator_
    lr_results, y_plot = evaluate_classifier(
        best_classifier_lr, X_test, y_test, X_ex, y_ex
    )
    lr_results["AUC"] = roc_auc_score(
        y_test, best_classifier_lr.predict_proba(X_test)[:, 1]
    )
    lr_results["AUC_ex"] = roc_auc_score(
        y_ex, best_classifier_lr.predict_proba(X_ex)[:, 1]
    )
    return lr_results, y_plot


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, X_ex, y_ex):
    svm_classifier = svm.SVC(random_state=42)
    parameters_svm = {"random_state": [42]}
    grid_search_svm = GridSearchCV(
        estimator=svm_classifier, param_grid=parameters_svm, cv=5
    )
    svm_cv = grid_search_svm.fit(X_train, y_train)
    best_classifier_svm = svm_cv.best_estimator_
    svm_results, y_plot = evaluate_classifier(
        best_classifier_svm, X_test, y_test, X_ex, y_ex
    )
    return svm_results, y_plot


def train_and_evaluate_tree(X_train, y_train, X_test, y_test, X_ex, y_ex):
    parameters_tree = {"random_state": [42]}
    tree = DecisionTreeClassifier(random_state=42)
    grid_search_tree = GridSearchCV(estimator=tree, param_grid=parameters_tree, cv=5)
    tree_cv = grid_search_tree.fit(X_train, y_train)
    best_classifier_tree = tree_cv.best_estimator_
    tree_results, y_plot = evaluate_classifier(
        best_classifier_tree, X_test, y_test, X_ex, y_ex
    )
    return tree_results, y_plot


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, X_ex, y_ex):
    parameters_knn = {}
    knn = KNeighborsClassifier()
    grid_search_knn = GridSearchCV(estimator=knn, param_grid=parameters_knn, cv=5)
    knn_cv = grid_search_knn.fit(X_train, y_train)
    best_classifier_knn = knn_cv.best_estimator_
    knn_results, y_plot = evaluate_classifier(
        best_classifier_knn, X_test, y_test, X_ex, y_ex
    )
    return knn_results, y_plot


def train_and_evaluate_forest(X_train, y_train, X_test, y_test, X_ex, y_ex):
    parameters_forest = {"random_state": [42]}
    forest = RandomForestClassifier(random_state=42)
    grid_search_forest = GridSearchCV(
        estimator=forest, param_grid=parameters_forest, cv=5
    )
    forest_cv = grid_search_forest.fit(X_train, y_train)
    best_classifier_forest = forest_cv.best_estimator_
    forest_results, y_plot = evaluate_classifier(
        best_classifier_forest, X_test, y_test, X_ex, y_ex
    )
    return forest_results, y_plot


def train_and_evaluate_neural(X_train, y_train, X_test, y_test, X_ex, y_ex):
    parameters_neural = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["relu"],  # Different activation functions
        "solver": ["adam"],  # Different solvers
        "alpha": [0.0001],  # Regularization strength
        "learning_rate": ["invscaling"],  # Learning rate schedule
        "max_iter": [200],  # Maximum number of iterations
        "random_state": [42],
        "early_stopping": [True, False],
        "n_iter_no_change": [
            10
        ],  # Number of iterations with no improvement to wait before stopping
    }
    neural = MLPClassifier(random_state=42)
    grid_search_neural = GridSearchCV(
        estimator=neural, param_grid=parameters_neural, cv=5
    )
    neural_cv = grid_search_neural.fit(X_train, y_train)
    best_classifier_neural = neural_cv.best_estimator_
    neural_results, y_plot = evaluate_classifier(
        best_classifier_neural, X_test, y_test, X_ex, y_ex
    )
    return neural_results, y_plot


def train_and_evaluate_boost(X_train, y_train, X_test, y_test, X_ex, y_ex):
    parameters_boost = {"random_state": [42]}
    boost = XGBClassifier(random_state=42)
    grid_search_boost = GridSearchCV(estimator=boost, param_grid=parameters_boost, cv=5)
    boost_cv = grid_search_boost.fit(X_train, y_train)
    best_classifier_boost = boost_cv.best_estimator_
    boost_results, y_plot = evaluate_classifier(
        best_classifier_boost, X_test, y_test, X_ex, y_ex
    )
    return boost_results, y_plot


# # Final results


list_Outcome_InhospitalMortality = []


y_train_ = np.array(y_train)
y_test_ = np.array(y_test)
X_test = np.array(X_test)
y_ex_ = np.array(y_ex["Outcome_InhospitalMortality"].tolist())
X_ex = np.array(X_ex)


list_Outcome_InhospitalMortality.extend(
    [
        logistic_regression_classifier(X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_svm(X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_tree(X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_knn(X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_forest(X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_neural(X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
        train_and_evaluate_boost(X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
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

print(df)
# df.to_excel('CML.xlsx',index=False)
