# Data Mining: course project - Oct 2024 - Mohammad Babazadeh (s4474507)


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, make_scorer

# set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def preprocess_data(train_data_path):
    """Preprocess the training data and return features and target."""
    df_train = pd.read_csv(train_data_path)

    # separate features (X) and target (y)
    X = df_train.iloc[:, :-1]
    y = df_train.iloc[:, -1]

    # handle missing values
    num_imputer = SimpleImputer(strategy="mean")
    X.iloc[:, :103] = num_imputer.fit_transform(X.iloc[:, :103])

    nom_imputer = SimpleImputer(strategy="most_frequent")
    X.iloc[:, 103:105] = nom_imputer.fit_transform(X.iloc[:, 103:105])

    # normalize numerical columns
    scaler = MinMaxScaler()
    X.iloc[:, :103] = scaler.fit_transform(X.iloc[:, :103])

    return X, y, num_imputer, nom_imputer, scaler


def select_features(X, y):
    """Select features using SelectKBest."""
    selector = SelectKBest(f_classif, k=50)
    X_selected = selector.fit_transform(X, y)
    print(f"Number of selected features: {X_selected.shape[1]}")
    return X_selected, selector


def train_models(X_selected, y):
    """Train machine learning models and return the best one."""
    param_grids = {
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'k-NN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    }

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED),
        'k-NN': KNeighborsClassifier()
    }

    # stratified 10-Fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    f1_scorer = make_scorer(f1_score)

    best_model = None
    best_model_name = None
    best_f1_score = 0
    best_accuracy = 0

    for model_name, model in models.items():
        print(f"Tuning {model_name}...")
        search = RandomizedSearchCV(
            model,
            param_grids[model_name],
            n_iter=20,
            cv=cv,
            scoring=f1_scorer,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        search.fit(X_selected, y)

        # get the best model and its f1-score
        if search.best_score_ > best_f1_score:
            best_f1_score = search.best_score_
            best_model = search.best_estimator_
            best_model_name = model_name

            # calculate accuracy during the 10-fold CV
            best_accuracy = cross_val_score(
                best_model,
                X_selected,
                y,
                cv=cv,
                scoring='accuracy'
            ).mean()

        print(f"Best {model_name} parameters: {search.best_params_}")
        print(f"Best {model_name} F1 score: {search.best_score_:.4f}\n")

    print(
        f"Selected best model: {best_model_name}"
        f"with F1 score: {best_f1_score:.4f}"
        f"and accuracy: {best_accuracy:.4f}"
    )
    return best_model, best_accuracy, best_f1_score


def preprocess_test_data(test_data_path, num_imputer, nom_imputer, scaler):
    """Preprocess the test data."""
    df_test = pd.read_csv(test_data_path)

    # handle missing values and normalise test data
    df_test.iloc[:, :103] = num_imputer.transform(df_test.iloc[:, :103])
    df_test.iloc[:, 103:105] = nom_imputer.transform(df_test.iloc[:, 103:105])
    df_test.iloc[:, :103] = scaler.transform(df_test.iloc[:, :103])

    return df_test


def main():
    # file paths
    train_data_path = 'DM_Project_24.csv'
    test_data_path = 'test_data.csv'

    # preprocess training data
    X, y, num_imputer, nom_imputer, scaler = preprocess_data(train_data_path)

    # select features
    X_selected, selector = select_features(X, y)

    # train models and get the best one
    best_model, best_accuracy, best_f1_score = train_models(X_selected, y)

    # preprocess test data
    df_test = preprocess_test_data(
        test_data_path,
        num_imputer,
        nom_imputer,
        scaler
    )

    # feature selection for test data (same as training dataset)
    df_test_selected = selector.transform(df_test)
    print(f"Number of selected features in test data: "
          f"{df_test_selected.shape[1]}")

    # train the best model on the full training data
    best_model.fit(X_selected, y)

    # apply the best trained model to the test data
    test_predictions = best_model.predict(df_test_selected)

    # save the labels in the result report
    result_report = list(test_predictions)
    result_report.append(
        f"{round(best_accuracy, 3)},{round(best_f1_score, 3)}"
    )

    result_file_path = 's4474507.infs4203'
    with open(result_file_path, 'w') as f:
        for item in result_report:
            f.write(f"{item} \n")

    print(f"Result report saved to {result_file_path}")


if __name__ == "__main__":
    main()
