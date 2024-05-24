
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# loading data
df = pd.read_excel("3.xlsx")


# Drop column: 'L1_BloodGroup_First'
df = df.drop(columns=['L1_BloodGroup_First'])
# Drop rows with missing data in column: 'Outcome_InhospitalMortality'
df = df.dropna(subset=['Outcome_InhospitalMortality'])


# splitting internal and external validation

df_internal_validation = df[df['Patient_Hospital'] != "Hospital2"]
df_external_validation = df[df['Patient_Hospital'] == "Hospital2"]
df_internal_validation = df_internal_validation.drop(
    columns=['Patient_Hospital'])
df_external_validation = df_external_validation.drop(
    columns=['Patient_Hospital'])


# definig features and target variables

X = df_internal_validation.drop(columns=[
                                'Outcome_InhospitalMortality', 'TM_S_Intubation', 'Outcome_ICUadmission', 'TM_S_Dialysis'])
y = df_internal_validation['Outcome_InhospitalMortality']


# definig features and target variables for extrenal validation dataset

X_ex = df_external_validation.drop(columns=[
                                   'Outcome_InhospitalMortality', 'TM_S_Intubation', 'Outcome_ICUadmission', 'TM_S_Dialysis'])
y_ex = df_external_validation['Outcome_InhospitalMortality']


# train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# indicating numerical variables

num = [
    "symtpm_to_referral",
    "VS_O2satwithoutsupp",
    "VS_PR",
    "VS_diastolic BP",
    "VS_Systolic BP",
    "VS_RR",
    "VS_T",
    "LAB_WBC_1",
    "LAB_LYMPHH_1",
    "LAB_NEUT_1",
    "LAB_PLT_1",
    "LAB_HB_1",
    "LAB_MCV_1",
    "LAB_CR_1",
    "LAB_NA_First",
    "LAB_K_First",
    "LAB_ALKP_First",
    "LAB_ESR_First",
    "LAB_CPK_First",
    "LAB_PTT_First",
    "LAB_PT_First",
    "LAB_INR_First",
    "Demographic_Age"
]


# indicationg categorical variables

cat = X.drop(columns=num).columns.tolist()


# imputing categorical data with knn imputer

imputer = KNNImputer(n_neighbors=5)

X_train_cat = imputer.fit_transform(np.array(X_train[cat]))
X_test_cat = imputer.fit_transform(np.array(X_test[cat]))
X_ex_cat = imputer.fit_transform(np.array(X_ex[cat]))

X_train_cat = pd.DataFrame(X_train_cat, columns=cat)
X_test_cat = pd.DataFrame(X_test_cat, columns=cat)
X_ex_cat = pd.DataFrame(X_ex_cat, columns=cat)


# imputing numerical data with terative imputer

imputer = IterativeImputer(max_iter=50, random_state=0)

X_train_num = imputer.fit_transform(X_train[num])
X_test_num = imputer.fit_transform(X_test[num])
X_ex_num = imputer.fit_transform(X_ex[num])

X_train_num = pd.DataFrame(X_train_num, columns=num)
X_test_num = pd.DataFrame(X_test_num, columns=num)
X_ex_num = pd.DataFrame(X_ex_num, columns=num)


# merging categorical and numerical data

X_train = pd.concat([X_train_cat, X_train_num], axis=1)
X_test = pd.concat([X_test_cat, X_test_num], axis=1)
X_ex = pd.concat([X_ex_cat, X_ex_num], axis=1)


# handing skewness in the dataset


def find_skewness(train, num):
    """
    Calculate the skewness of the columns and segregate the positive
    and negative skewed data.
    """
    skew_dict = {}
    for col in num:
        skew_dict[col] = train[col].skew()
    skew_dict = dict(sorted(skew_dict.items(), key=itemgetter(1)))
    positive_skew_dict = {k: v for (k, v) in skew_dict.items() if v > 0}
    negative_skew_dict = {k: v for (k, v) in skew_dict.items() if v < 0}
    return skew_dict, positive_skew_dict, negative_skew_dict


def add_constant(data, highly_pos_skewed):
    """
    Look for zeros in the columns. If zeros are present then the log(0) would result in -infinity.
    So before transforming it we need to add it with some constant.
    """
    C = 1
    for col in highly_pos_skewed.keys():
        if (col != 'SalePrice'):
            if (len(data[data[col] == 0]) > 0):
                data[col] = data[col] + C
    return data


def log_transform(data, highly_pos_skewed):
    """
    Log transformation of highly positively skewed columns.
    """
    for col in highly_pos_skewed.keys():
        if (col != 'SalePrice'):
            data[col] = np.log10(data[col])
    return data


def sqrt_transform(data, moderately_pos_skewed):
    """
    Square root transformation of moderately skewed columns.
    """
    for col in moderately_pos_skewed.keys():
        if (col != 'SalePrice'):
            data[col] = np.sqrt(data[col])
    return data


def reflect_sqrt_transform(data, moderately_neg_skewed):
    """
    Reflection and log transformation of highly negatively skewed 
    columns.
    """
    for col in moderately_neg_skewed.keys():
        if (col != 'SalePrice'):
            K = max(data[col]) + 1
            data[col] = np.sqrt(K - data[col])
    return data


"""
If skewness is less than -1 or greater than 1, the distribution is highly skewed.
If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
"""
skew_dict, positive_skew_dict, negative_skew_dict = find_skewness(X_train, num)
moderately_pos_skewed = {
    k: v for (k, v) in positive_skew_dict.items() if v > 0.5 and v <= 1}
highly_pos_skewed = {k: v for (k, v) in positive_skew_dict.items() if v > 1}
moderately_neg_skewed = {
    k: v for (k, v) in negative_skew_dict.items() if v > -1 and v <= 0.5}
highly_neg_skewed = {k: v for (k, v) in negative_skew_dict.items() if v < -1}
'''Transform train data.'''
X_train = add_constant(X_train, highly_pos_skewed)
X_train = log_transform(X_train, highly_pos_skewed)
X_train = sqrt_transform(X_train, moderately_pos_skewed)
X_train = reflect_sqrt_transform(X_train, moderately_neg_skewed)
'''Transform test data.'''
X_test = add_constant(X_test, highly_pos_skewed)
X_test = log_transform(X_test, highly_pos_skewed)
X_test = sqrt_transform(X_test, moderately_pos_skewed)
X_test = reflect_sqrt_transform(X_test, moderately_neg_skewed)

X_ex = add_constant(X_ex, highly_pos_skewed)
X_ex = log_transform(X_ex, highly_pos_skewed)
X_ex = sqrt_transform(X_ex, moderately_pos_skewed)
X_ex = reflect_sqrt_transform(X_ex, moderately_neg_skewed)


# handilng skewness can make some null values.
# here we handle these missing values

def clean_data(X_train):
    # Replace missing values with the mean of each column in: 'L1_BloodGroup_First', 'Demographic_Gender' and 74 other columns
    X_train = X_train.fillna({'Demographic_Gender': X_train['Demographic_Gender'].mean(), 'Symptom_Caugh': X_train['Symptom_Caugh'].mean(), 'Symptom_Dyspnea': X_train['Symptom_Dyspnea'].mean(), 'Symptom_Fever': X_train['Symptom_Fever'].mean(), 'Symptom_Chiver': X_train['Symptom_Chiver'].mean(), 'Symptom_Mylagia': X_train['Symptom_Mylagia'].mean(), 'Symptom_Weakness': X_train['Symptom_Weakness'].mean(), 'Symptom_LOC': X_train['Symptom_LOC'].mean(), 'Symptom_Sore through': X_train['Symptom_Sore through'].mean(), 'Symptom_Rhinorrhea': X_train['Symptom_Rhinorrhea'].mean(), 'Symptom_Smelling disorder': X_train['Symptom_Smelling disorder'].mean(), 'Symptom_nauseaVomit': X_train['Symptom_nauseaVomit'].mean(), 'Symptom_Anorexia': X_train['Symptom_Anorexia'].mean(), 'Symptom_Diarhhea': X_train['Symptom_Diarhhea'].mean(), 'Symptom_ChestPain': X_train['Symptom_ChestPain'].mean(), 'Symptom_Seizure': X_train['Symptom_Seizure'].mean(), 'Symptom_SkinLesion': X_train['Symptom_SkinLesion'].mean(), 'Symptom_Jointpain': X_train['Symptom_Jointpain'].mean(), 'Symptom_Headache': X_train['Symptom_Headache'].mean(), 'Symptom_AbdominalPain': X_train['Symptom_AbdominalPain'].mean(), 'Symptom_Earpain': X_train['Symptom_Earpain'].mean(), 'Symptom_Hemorrhasia': X_train['Symptom_Hemorrhasia'].mean(), 'Symptom_Hemiparesia': X_train['Symptom_Hemiparesia'].mean(), 'MH_PregnanAcy': X_train['MH_PregnanAcy'].mean(), 'MH_CurremtSmoker': X_train['MH_CurremtSmoker'].mean(), 'MH_Alcoholuser': X_train['MH_Alcoholuser'].mean(), 'MH_Opiumuser': X_train['MH_Opiumuser'].mean(), 'MH_Hookahuser': X_train['MH_Hookahuser'].mean(), 'MH_HTN': X_train['MH_HTN'].mean(), 'MH_IHD': X_train['MH_IHD'].mean(), 'MH_CABG': X_train['MH_CABG'].mean(), 'MH_CHF': X_train['MH_CHF'].mean(), 'MH_Ashtma': X_train['MH_Ashtma'].mean(), 'MH_COPD': X_train['MH_COPD'].mean(), 'MH_DM': X_train['MH_DM'].mean(
    ), 'MH_Pneumonia': X_train['MH_Pneumonia'].mean(), 'MH_CVA': X_train['MH_CVA'].mean(), 'MH_GIdisorder': X_train['MH_GIdisorder'].mean(), 'MH_CKD': X_train['MH_CKD'].mean(), 'MH_RA': X_train['MH_RA'].mean(), 'Cancer': X_train['Cancer'].mean(), 'MH_HLP': X_train['MH_HLP'].mean(), 'MH_Hep C': X_train['MH_Hep C'].mean(), 'MH_Thyroid dysfunction': X_train['MH_Thyroid dysfunction'].mean(), 'MH_Immunocompromised': X_train['MH_Immunocompromised'].mean(), 'MH_ChronicSeizure': X_train['MH_ChronicSeizure'].mean(), 'MH_TB': X_train['MH_TB'].mean(), 'MH_Anemia': X_train['MH_Anemia'].mean(), 'MH_Fattyliver': X_train['MH_Fattyliver'].mean(), 'MH_Psychologicaldisorder': X_train['MH_Psychologicaldisorder'].mean(), 'MH_Parkinson': X_train['MH_Parkinson'].mean(), 'MH_Alzhimer': X_train['MH_Alzhimer'].mean(), 'symtpm_to_referral': X_train['symtpm_to_referral'].mean(), 'VS_O2satwithoutsupp': X_train['VS_O2satwithoutsupp'].mean(), 'VS_PR': X_train['VS_PR'].mean(), 'VS_diastolic BP': X_train['VS_diastolic BP'].mean(), 'VS_Systolic BP': X_train['VS_Systolic BP'].mean(), 'VS_RR': X_train['VS_RR'].mean(), 'VS_T': X_train['VS_T'].mean(), 'LAB_WBC_1': X_train['LAB_WBC_1'].mean(), 'LAB_LYMPHH_1': X_train['LAB_LYMPHH_1'].mean(), 'LAB_NEUT_1': X_train['LAB_NEUT_1'].mean(), 'LAB_PLT_1': X_train['LAB_PLT_1'].mean(), 'LAB_HB_1': X_train['LAB_HB_1'].mean(), 'LAB_MCV_1': X_train['LAB_MCV_1'].mean(), 'LAB_CR_1': X_train['LAB_CR_1'].mean(), 'LAB_NA_First': X_train['LAB_NA_First'].mean(), 'LAB_K_First': X_train['LAB_K_First'].mean(), 'LAB_ALKP_First': X_train['LAB_ALKP_First'].mean(), 'LAB_ESR_First': X_train['LAB_ESR_First'].mean(), 'LAB_CPK_First': X_train['LAB_CPK_First'].mean(), 'LAB_PTT_First': X_train['LAB_PTT_First'].mean(), 'LAB_PT_First': X_train['LAB_PT_First'].mean(), 'LAB_INR_First': X_train['LAB_INR_First'].mean(), 'Demographic_Age': X_train['Demographic_Age'].mean()})
    return X_train


X_train = clean_data(X_train.copy())
X_test = clean_data(X_test.copy())
X_ex = clean_data(X_ex.copy())


# define the function to standard numerca data

transform = StandardScaler()


def standard(f):
    df_ex_normalize = transform.fit_transform(f)
    df_ex_normalize = pd.DataFrame(df_ex_normalize)
    df_ex_normalize.columns = f.columns
    return df_ex_normalize


# # standarding nmerical data with StandardScaler
num = list(X_train.columns[X_train.columns.get_loc('VS_O2satwithoutsupp'): X_train.columns.get_loc('Demographic_Age') + 1]
           )

X_train[num] = standard(X_train[num])
X_test[num] = standard(X_test[num])
X_ex[num] = standard(X_ex[num])


# we tried different undersampling and oversampling techniques. SMOTE performed better than others.

over_sampler = SMOTE()
X_train, y_train = over_sampler.fit_resample(X_train, y_train)
X_test, y_test = over_sampler.fit_resample(X_test, y_test)
X_ex, y_ex = over_sampler.fit_resample(X_ex, y_ex)


# function that selects most important features using lasso

def lasso_feature_selector(X_train, y_train, X_test, X_ex):

    lasso = Lasso(alpha=0.001, random_state=42)
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
    return X_train_selected_Mortality_ICU, X_test_selected_Mortality_ICU, X_ex, selected_feature_indices


X_train, X_test, X_ex, selected_feature_indices = lasso_feature_selector(
    X_train, y_train, X_test, X_ex)


# machine learning models


# metrics caculator function


def calculate_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    cm = confusion_matrix(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    specificity = cm[0, 0] / \
        (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
    f1 = f1_score(y_test, y_predicted)

    return accuracy, precision, recall, specificity, f1


def calculate_and_update_model_metrics(y_test, y_predicted, y_ex, y_ex_predicted):
    accuracy, precision, recall, specificity, f1 = calculate_metrics(
        y_test, y_predicted)
    model = {}
    model["accuracy"] = accuracy
    model["precision"] = precision
    model["recall"] = recall
    model["specificity"] = specificity
    model["f1"] = f1
    model["AUC"] = roc_auc_score(y_test, y_predicted)

    accuracy_ex, precision_ex, recall_ex, specificity_ex, f1_ex = calculate_metrics(
        y_ex, y_ex_predicted)
    model["accuracy_ex"] = accuracy_ex
    model["precision_ex"] = precision_ex
    model["recall_ex"] = recall_ex
    model["specificity_ex"] = specificity_ex
    model["f1_ex"] = f1_ex
    model["AUC_ex"] = roc_auc_score(y_ex, y_ex_predicted)
    return model


def logistic_regression_classifier(X_train, y_train, X_test, y_test, X_ex, y_ex):
    parameters_lr = {'random_state': [42]}
    lr = LogisticRegression(random_state=42)
    grid_search_lr = GridSearchCV(
        estimator=lr,
        param_grid=parameters_lr,
        cv=5,
        n_jobs=-1
    )

    logreg_cv = grid_search_lr.fit(X_train, y_train)
    best_classifier_lr = logreg_cv.best_estimator_

    # Predictions and probabilities for test set
    y_pred_test = best_classifier_lr.predict(X_test)
    y_pred_proba_test = best_classifier_lr.predict_proba(X_test)[:, 1]

    # Predictions and probabilities for external set
    y_pred_external = best_classifier_lr.predict(X_ex)
    y_pred_proba_external = best_classifier_lr.predict_proba(X_ex)[:, 1]

    # Calculate metrics
    model_metrics = calculate_and_update_model_metrics(
        y_test, y_pred_test, y_ex, y_pred_external)

    y_plot = {
        "y_true": y_test,
        "y_predicted": y_pred_test,
        "y_true_ex": y_ex,
        "y_predicted_ex": y_pred_external,
        "y_pred_proba_test": y_pred_proba_test,
        "y_pred_proba_external": y_pred_proba_external
    }

    return model_metrics, y_plot


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_lr = {'random_state': [42], 'probability': [True]}
    lr = svm.SVC(random_state=42)
    grid_search_lr = GridSearchCV(
        estimator=lr,
        param_grid=parameters_lr,
        cv=5,
        n_jobs=-1
    )

    logreg_cv = grid_search_lr.fit(X_train, y_train)

    best_classifier_lr = logreg_cv.best_estimator_

    y_pred_test = best_classifier_lr.predict(X_test)
    y_pred_proba_test = best_classifier_lr.predict_proba(X_test)[:, 1]

    y_pred_external = best_classifier_lr.predict(X_ex)
    y_pred_proba_external = best_classifier_lr.predict_proba(X_ex)[:, 1]

    Logistic_Regression = Logistic_Regression = calculate_and_update_model_metrics(
        y_test, y_pred_test, y_ex, y_pred_external)

    y_plot = {
        "y_true": y_test,
        "y_predicted": y_pred_test,
        "y_true_ex": y_ex,
        "y_predicted_ex": y_pred_external,
        "y_pred_proba_test": y_pred_proba_test,
        "y_pred_proba_external": y_pred_proba_external
    }

    Logistic_Regression = (Logistic_Regression, y_plot)

    return Logistic_Regression


def train_and_evaluate_tree(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_lr = {'random_state': [42]}
    lr = DecisionTreeClassifier(random_state=42)
    grid_search_lr = GridSearchCV(
        estimator=lr,
        param_grid=parameters_lr,
        cv=5,
        n_jobs=-1
    )

    logreg_cv = grid_search_lr.fit(X_train, y_train)

    best_classifier_lr = logreg_cv.best_estimator_

    y_pred_test = best_classifier_lr.predict(X_test)
    y_pred_proba_test = best_classifier_lr.predict_proba(X_test)[:, 1]

    y_pred_external = best_classifier_lr.predict(X_ex)
    y_pred_proba_external = best_classifier_lr.predict_proba(X_ex)[:, 1]

    Logistic_Regression = Logistic_Regression = calculate_and_update_model_metrics(
        y_test, y_pred_test, y_ex, y_pred_external)

    y_plot = {
        "y_true": y_test,
        "y_predicted": y_pred_test,
        "y_true_ex": y_ex,
        "y_predicted_ex": y_pred_external,
        "y_pred_proba_test": y_pred_proba_test,
        "y_pred_proba_external": y_pred_proba_external
    }

    Logistic_Regression = (Logistic_Regression, y_plot)

    return Logistic_Regression


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_lr = {}
    lr = KNeighborsClassifier()
    grid_search_lr = GridSearchCV(
        estimator=lr,
        param_grid=parameters_lr,
        cv=5,
        n_jobs=-1
    )

    logreg_cv = grid_search_lr.fit(X_train, y_train)

    best_classifier_lr = logreg_cv.best_estimator_

    y_pred_test = best_classifier_lr.predict(X_test)
    y_pred_proba_test = best_classifier_lr.predict_proba(X_test)[:, 1]

    y_pred_external = best_classifier_lr.predict(X_ex)
    y_pred_proba_external = best_classifier_lr.predict_proba(X_ex)[:, 1]

    Logistic_Regression = calculate_and_update_model_metrics(
        y_test, y_pred_test, y_ex, y_pred_external)

    y_plot = {
        "y_true": y_test,
        "y_predicted": y_pred_test,
        "y_true_ex": y_ex,
        "y_predicted_ex": y_pred_external,
        "y_pred_proba_test": y_pred_proba_test,
        "y_pred_proba_external": y_pred_proba_external
    }

    Logistic_Regression = (Logistic_Regression, y_plot)

    return Logistic_Regression


def train_and_evaluate_forest(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_lr = {'random_state': [42]}
    lr = RandomForestClassifier(random_state=42)
    grid_search_lr = GridSearchCV(
        estimator=lr,
        param_grid=parameters_lr,
        cv=5,
        n_jobs=-1
    )

    logreg_cv = grid_search_lr.fit(X_train, y_train)

    best_classifier_lr = logreg_cv.best_estimator_

    y_pred_test = best_classifier_lr.predict(X_test)
    y_pred_proba_test = best_classifier_lr.predict_proba(X_test)[:, 1]

    y_pred_external = best_classifier_lr.predict(X_ex)
    y_pred_proba_external = best_classifier_lr.predict_proba(X_ex)[:, 1]

    Logistic_Regression = Logistic_Regression = calculate_and_update_model_metrics(
        y_test, y_pred_test, y_ex, y_pred_external)

    y_plot = {
        "y_true": y_test,
        "y_predicted": y_pred_test,
        "y_true_ex": y_ex,
        "y_predicted_ex": y_pred_external,
        "y_pred_proba_test": y_pred_proba_test,
        "y_pred_proba_external": y_pred_proba_external
    }

    Logistic_Regression = (Logistic_Regression, y_plot)

    return Logistic_Regression


def train_and_evaluate_boost(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_lr = {'random_state': [42]}
    lr = XGBClassifier(random_state=42)
    grid_search_lr = GridSearchCV(
        estimator=lr,
        param_grid=parameters_lr,
        cv=5,
        n_jobs=-1
    )

    logreg_cv = grid_search_lr.fit(X_train, y_train)

    best_classifier_lr = logreg_cv.best_estimator_

    y_pred_test = best_classifier_lr.predict(X_test)
    y_pred_proba_test = best_classifier_lr.predict_proba(X_test)[:, 1]

    y_pred_external = best_classifier_lr.predict(X_ex)
    y_pred_proba_external = best_classifier_lr.predict_proba(X_ex)[:, 1]

    Logistic_Regression = calculate_and_update_model_metrics(
        y_test, y_pred_test, y_ex, y_pred_external)

    y_plot = {
        "y_true": y_test,
        "y_predicted": y_pred_test,
        "y_true_ex": y_ex,
        "y_predicted_ex": y_pred_external,
        "y_pred_proba_test": y_pred_proba_test,
        "y_pred_proba_external": y_pred_proba_external
    }

    Logistic_Regression = (Logistic_Regression, y_plot)

    return Logistic_Regression


def train_and_evaluate_neural(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_neural = {
        # You can adjust the architecture here
        'hidden_layer_sizes': [(100,), (50, 50)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'max_iter': [200],
        'random_state': [42],
        'early_stopping': [True],
        'validation_fraction': [0.1],
        'n_iter_no_change': [10]
    }

    neural = MLPClassifier(random_state=42)

    grid_search_neural = GridSearchCV(
        estimator=neural,
        param_grid=parameters_neural,
        cv=5
    )

    neural_cv = grid_search_neural.fit(X_train, y_train)

    best_classifier_lr = neural_cv.best_estimator_

    y_pred_test = best_classifier_lr.predict(X_test)
    y_pred_proba_test = best_classifier_lr.predict_proba(X_test)[:, 1]

    y_pred_external = best_classifier_lr.predict(X_ex)
    y_pred_proba_external = best_classifier_lr.predict_proba(X_ex)[:, 1]

    Logistic_Regression = calculate_and_update_model_metrics(
        y_test, y_pred_test, y_ex, y_pred_external)

    y_plot = {
        "y_true": y_test,
        "y_predicted": y_pred_test,
        "y_true_ex": y_ex,
        "y_predicted_ex": y_pred_external,
        "y_pred_proba_test": y_pred_proba_test,
        "y_pred_proba_external": y_pred_proba_external
    }

    Logistic_Regression = (Logistic_Regression, y_plot)

    return Logistic_Regression


def train_and_evaluate_neural(X_train, y_train, X_test, y_test, X_ex, y_ex):

    parameters_neural = {
        # You can adjust the architecture here
        'hidden_layer_sizes': [(100,), (50, 50)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'max_iter': [200],
        'random_state': [42],
        'early_stopping': [True],
        'validation_fraction': [0.1],
        'n_iter_no_change': [10]
    }

    lr = MLPClassifier(random_state=42)

    grid_search_lr = GridSearchCV(
        estimator=lr,
        param_grid=parameters_neural,
        # cv=5
    )

    logreg_cv = grid_search_lr.fit(X_train, y_train)

    best_classifier_lr = logreg_cv.best_estimator_

    y_pred_test = best_classifier_lr.predict(X_test)
    y_pred_proba_test = best_classifier_lr.predict_proba(X_test)[:, 1]

    y_pred_external = best_classifier_lr.predict(X_ex)
    y_pred_proba_external = best_classifier_lr.predict_proba(X_ex)[:, 1]

    Logistic_Regression = calculate_and_update_model_metrics(
        y_test, y_pred_test, y_ex, y_pred_external)

    y_plot = {
        "y_true": y_test,
        "y_predicted": y_pred_test,
        "y_true_ex": y_ex,
        "y_predicted_ex": y_pred_external,
        "y_pred_proba_test": y_pred_proba_test,
        "y_pred_proba_external": y_pred_proba_external
    }

    Logistic_Regression = (Logistic_Regression, y_plot)

    return Logistic_Regression


# final function that shows the results of all models in one table


def run_all_models(X_train, y_train, X_test, y_test, X_ex, y_ex):
    list_Outcome_InhospitalMortality = []
    y_train_ = np.array(y_train)
    y_test_ = np.array(y_test)

    y_ex_ = np.array(y_ex)

    list_Outcome_InhospitalMortality.extend([logistic_regression_classifier(X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
                                             train_and_evaluate_svm(
                                                 X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
                                            train_and_evaluate_tree(
                                                X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
                                            train_and_evaluate_knn(
                                                X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
                                            train_and_evaluate_forest(
                                                X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
                                            train_and_evaluate_boost(
                                                X_train, y_train_, X_test, y_test_, X_ex, y_ex_),
                                            train_and_evaluate_neural(X_train, y_train, X_test, y_test, X_ex, y_ex)])

    result_dic_list_Outcome_InhospitalMortality = dict(zip(
        ['logistic regression', 'SVM', 'Decision tree', 'knn', 'Random forest', 'XGboost', 'neural net'], list_Outcome_InhospitalMortality))

    merged_dict = {}

    # List of dictionary names and their corresponding dictionaries
    dict_list = [('Outcome_InhospitalMortality',
                  result_dic_list_Outcome_InhospitalMortality)]

    # Merge the dictionaries
    for name, result_dict in dict_list:
        merged_dict[name] = result_dict

    # The merged_dict now contains all the dictionaries merged together

    dff = pd.DataFrame(merged_dict)

    dff = dff.transpose()

    dff.reset_index(inplace=True)
    dff.rename(columns={'index': 'Method'}, inplace=True)

    data = []

    for outcome, models in merged_dict.items():
        for model, metrics in models.items():
            accuracy, precision, recall, specificity, f1, AUC, accuracy_ex, precision_ex, recall_ex, specificity_ex, f1_ex, AUC_ex = metrics[0]['accuracy'], metrics[0]['precision'], metrics[0]['recall'], metrics[0][
                'specificity'], metrics[0]['f1'], metrics[0]['AUC'], metrics[0]['accuracy_ex'], metrics[0]['precision_ex'], metrics[0]['recall_ex'], metrics[0]['specificity_ex'], metrics[0]['f1_ex'], metrics[0]['AUC_ex']
            y_true, y_predicted, y_true_ex, y_predicted_ex, y_pred_proba_test, y_pred_proba_external = metrics[1]['y_true'], metrics[1][
                'y_predicted'], metrics[1]['y_true_ex'], metrics[1]['y_predicted_ex'], metrics[1]['y_pred_proba_test'], metrics[1]['y_pred_proba_external']
            data.append([outcome, model, accuracy, precision, recall, specificity, f1, AUC, accuracy_ex, precision_ex, recall_ex, specificity_ex, f1_ex, AUC_ex,
                        y_true.tolist(), y_predicted.tolist(), y_true_ex.tolist(), y_predicted_ex.tolist(), y_pred_proba_test.tolist(), y_pred_proba_external.tolist()])

    columns = ['Outcome', 'Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'AUC', 'Accuracy_ex', 'Precision_ex', 'Recall_ex',
               'Specificity_ex', 'F1_ex', 'AUC_ex', 'y_true', 'y_predicted', 'y_true_ex', 'y_predicted_ex', 'y_pred_proba_test', 'y_pred_proba_external']

    dff = pd.DataFrame(data, columns=columns)

    return dff


# saving the result's table
d = run_all_models(X_train, y_train, X_test, y_test, X_ex, y_ex)


# d.to_csv('CML_prob.csv',index=False)
d.to_excel('CML_prob.xlsx', index=False)
