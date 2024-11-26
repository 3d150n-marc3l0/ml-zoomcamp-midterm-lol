import os
import yaml
import argparse

import pandas as pd
import numpy as np
# Logging
import logging
# Files
import json
import pickle 
# Plot
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# Optimization
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope

# Crear un logger
logger = logging.getLogger()

# Configuración para el archivo
log_file_path = os.path.join("logs",'train-logs.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)  # Puedes ajustar el nivel aquí (INFO, DEBUG, etc.)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# Configuración para la consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Igualmente, puedes ajustar el nivel de consola
console_handler.setFormatter(file_formatter)

# Agregar ambos handlers al logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Establecer el nivel de log para el logger
logger.setLevel(logging.INFO)  # Este es el nivel general del logger (puedes ajustarlo)


def read_dataset(raw_data_path):
    # 
    if not os.path.exists(raw_data_path):
        logger.error(f"File doesn't exists: {raw_data_path}")
    raw_data = data_raw = pd.read_csv(raw_data_path)
    return raw_data

def plot_precision_recall(metrics, output_file='precision_recall_plot.png'):
    thresholds = metrics['threshold'].values
    precisions = metrics['precision'].values
    recalls = metrics['recall'].values
    
    # Crear la figura y el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='green')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.title('Precision and Recall vs. Threshold')
    plt.legend()
    plt.grid(True)

    # Encontrar el umbral donde precision y recall se intersectan
    intersect_threshold_idx = np.where(np.abs(np.array(precisions) - np.array(recalls)) < 0.01)[0]
    intersect_threshold = thresholds[intersect_threshold_idx][0]

    # Dibujar la línea vertical en el punto de intersección
    plt.axvline(x=intersect_threshold, color='red', linestyle='--', label=f'Intersection at {intersect_threshold:.2f}')

    # Guardar el gráfico en un archivo (en vez de mostrarlo)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # Cerrar la figura para liberar recursos
    plt.close()

    # Imprimir el umbral de intersección
    print(f"Precision and recall intersect at threshold: {intersect_threshold:.3f}")


# Define the KDA calculation function
def lol_kda(kills, assists, deaths):
    kde = (kills + assists) / deaths.replace(0, 1)  # Avoid division by zero
    return kde


def run_data_wrangling(data_raw):
    # Assuming data_raw is a pandas DataFrame
    data_column_rowname = "gameId"
    
    # Convert the 'gameId' column into the row index and remove it from the columns
    data_clean = data_raw.set_index(data_column_rowname)
    
    # Remove the 'redFirstBlood' column from data_clean
    data_clean = data_clean.drop(columns=['redFirstBlood'])

    # Remove columns: blueAvgLevel and redAvgLevel from data_clean
    data_clean = data_clean.drop(columns=['blueAvgLevel', 'redAvgLevel'])

    # Remove columns: blueCSPerMin and redCSPerMin from data_clean
    data_clean = data_clean.drop(columns=['blueCSPerMin', 'redCSPerMin'])
    
    # Remove columns: blueGoldPerMin and redGoldPerMin from data_clean
    data_clean = data_clean.drop(columns=['blueGoldPerMin', 'redGoldPerMin'])

    # Drop columns: redGoldDiff, redExperienceDiff, blueTotalExperience, redTotalExperience, blueTotalGold, redTotalGold
    data_clean = data_clean.drop(columns=['redGoldDiff', 'redExperienceDiff', 'blueTotalExperience', 'redTotalExperience', 'blueTotalGold', 'redTotalGold'])

    # Drop columns: blueEliteMonsters and redEliteMonsters
    data_clean = data_clean.drop(columns=['blueEliteMonsters', 'redEliteMonsters'])
    
    # Calculate the KDA difference
    data_clean['blueKDADiff'] = lol_kda(data_clean['blueKills'], 
                                    data_clean['blueAssists'], 
                                    data_clean['blueDeaths']) - \
                                lol_kda(data_clean['redKills'], 
                                    data_clean['redAssists'], 
                                    data_clean['redDeaths'])

    # Drop the columns related to kills, assists, and deaths
    data_clean = data_clean.drop(columns=['blueKills', 'blueAssists', 'blueDeaths', 
                                      'redKills', 'redAssists', 'redDeaths'])

    # Calculate the difference between the teams' total minions killed
    data_clean['blueCSDiff'] = data_clean['blueTotalMinionsKilled'] - data_clean['redTotalMinionsKilled']

    # Calculate the difference between the teams' total jungle minions killed
    data_clean['blueJGCSDiff'] = data_clean['blueTotalJungleMinionsKilled'] - data_clean['redTotalJungleMinionsKilled']

    # Drop the columns no longer needed: blueTotalMinionsKilled, redTotalMinionsKilled,
    # blueTotalJungleMinionsKilled, and redTotalJungleMinionsKilled
    data_clean = data_clean.drop(columns=['blueTotalMinionsKilled', 'redTotalMinionsKilled', 
                                      'blueTotalJungleMinionsKilled', 'redTotalJungleMinionsKilled'])
    
    # Calculate the difference between the teams' wards placed
    data_clean['blueWardDiff'] = data_clean['blueWardsPlaced'] - data_clean['redWardsPlaced']
    
    # Calculate the difference between the teams' wards destroyed
    data_clean['blueDestWardDiff'] = data_clean['blueWardsDestroyed'] - data_clean['redWardsDestroyed']
    
    # Drop the columns no longer needed: blueWardsPlaced, redWardsPlaced,
    # blueWardsDestroyed, and redWardsDestroyed
    data_clean = data_clean.drop(columns=['blueWardsPlaced', 'redWardsPlaced', 
                                      'blueWardsDestroyed', 'redWardsDestroyed'])
    
    data_clean['blueFirstBlood'] = data_clean['blueFirstBlood'].astype('category')
    
    return data_clean



def run_partition_data(data, target, seed=42):
    df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=seed)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

    # Reset index
    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Y
    y_full_train = df_full_train[target].values
    y_train = df_train[target].values
    y_valid = df_val[target].values
    y_test = df_test[target].values

    # Remove y
    df_full_train = df_full_train.drop(columns=[target])
    df_train = df_train.drop(columns=[target])
    df_valid = df_val.drop(columns=[target])
    df_test = df_test.drop(columns=[target])

    return (
        df_full_train, y_full_train,
        df_train, y_train,
        df_valid, y_valid,
        df_test, y_test
    )

def feature_importance_extr(
    X_train, y_train, X_test, y_test,
    numerical_features, categorical_features
):
    # Preproessing
    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
    )
    scaler = StandardScaler()
    preprocessor = ColumnTransformer(transformers=[
        ('num', scaler, numerical_features),
        ("cat", categorical_encoder, categorical_features),
    ])

    # Train a Random Forest model
    classifier = ExtraTreesClassifier(
        n_estimators = 100, 
        criterion ='entropy', 
        min_samples_leaf = 5, 
        max_features = 5
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", classifier)
        ]
    )
    model.fit(X_train, y_train)
    
    print(f"ETR train accuracy: {model.score(X_train, y_train):.3f}")
    print(f"ETR test accuracy : {model.score(X_test, y_test):.3f}")

    # Getting the importance of the features
    feature_names = model[:-1].get_feature_names_out()
    importances = model[-1].feature_importances_
    
    # Crear un DataFrame con la importancia de las características
    importance_df = pd.Series(
        importances, index=feature_names
    ).sort_values(ascending=False)

    return importance_df

def run_feature_selection(
    df_train, y_train,
    df_test, y_test, 
    numerical_features, categorical_features
):
    # Feature importance
    extr_importance_df = feature_importance_extr(
        df_train, y_train, 
        df_test, y_test, 
        numerical_features, categorical_features
    )

    # Select Columns 
    selected_cols_extr = [tra_col.split('__')[1]  for tra_col in extr_importance_df[extr_importance_df > 0.03].index]
    print(f"selected_cols: {selected_cols_extr}")

    categorical_features_fs = [col for col in selected_cols_extr if col in categorical_features]
    numerical_features_fs = [col for col in selected_cols_extr if col not in categorical_features_fs]
    return numerical_features_fs, categorical_features_fs


def calculate_precision_recall_f1(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        '''
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        '''
        recall = recall_score(y_val, y_pred >= t)
        precision = precision_score(y_val, y_pred >= t)
        f1 = f1_score(y_val, y_pred >= t)

        scores.append((t, precision, recall, f1))

    columns = ['threshold', 'precision', 'recall', 'f1_score']
    df_scores = pd.DataFrame(scores, columns=columns)
    
    return df_scores

def evaluate_model(
    model,
    X_test, y_test
):
    # Evaluate 
    y_test_pred_proba = model.predict_proba(X_test)[::,1]

    # Calcualte metrics
    metrics_df = calculate_precision_recall_f1(y_test, y_test_pred_proba)

    return metrics_df

def train_rf(
    X_train, y_train, #X_test, y_test,
    X_valid, y_valid,
    numerical_features, categorical_features,
    params_rf
):
    # Select features
    X_train = X_train[numerical_features + categorical_features]
    X_valid = X_valid[numerical_features + categorical_features]
    print(f"train: {X_train.shape}")
    print(f"valid: {X_valid.shape}")
    
    # Preprocessing
    transformers = []
    # Imputing categorical features)
    if categorical_features and len(categorical_features) > 0:
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
        )
        transformers.append(('cat', SimpleImputer(strategy='constant', fill_value=0), categorical_features))
    # Scaling numerical features
    transformers.append(('num', StandardScaler(), numerical_features))
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Create a pipeline with preprocessing and classification
    classifier = RandomForestClassifier(**params_rf)

    # Train model
    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('classifier', classifier)
    ])
    model.fit(X_train, y_train)

    # Evaluate model
    y_train_pred_proba = model.predict_proba(X_train)[::,1]
    y_valid_pred_proba = model.predict_proba(X_valid)[::,1]

    #calculate AUC of model
    train_acc = accuracy_score(y_train, y_train_pred_proba >= 0.5)
    test_acc = accuracy_score(y_valid, y_valid_pred_proba >= 0.5)
    test_auc = roc_auc_score(y_valid, y_valid_pred_proba)
    
    #print AUC score
    print(f"RF train acc: {train_acc:.3f}")
    print(f"RF valid acc: {test_acc:.3f}")
    print(f"RF valid auc: {test_auc:.3f}")

    return {
        'model': model,
        'metrics': {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_auc': test_auc,
        }
    }


# Function to optimize the classification model
def optimize_rf(
    X_train, y_train,
    numerical_features, categorical_features, 
    max_evals=10
):
    
    # Select features
    X_train = X_train[numerical_features + categorical_features]
    n_features = len(numerical_features + categorical_features)
    print(f"train: {X_train.shape}")
    
    # 3. Hyperparameter optimization using Hyperopt
    def objective(params):
        # Update the pipeline with the optimized hyperparameters
        logger.info(f"params: {params}")
        logger.info(f"train: {X_train.shape}")
        
        # Preprocessing
        transformers = []
        # Imputing categorical features)
        if categorical_features and len(categorical_features) > 0:
            transformers.append(('cat', SimpleImputer(strategy='constant', fill_value=0), categorical_features))
        # Scaling numerical features
        transformers.append(('num', StandardScaler(), numerical_features))
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Create a pipeline with preprocessing and classification
        classifier = RandomForestClassifier(
            max_depth = params['max_depth'],
            min_samples_leaf = params['min_samples_leaf'],
            n_estimators = params['n_estimators'], 
            random_state=42
        )
    
        # Train model
        model = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('classifier', classifier)
        ])
        
        # Perform cross-validation to evaluate the model
        scoring=['accuracy', #'precision_macro', 'recall_macro', 'f1_macro', 
                 'roc_auc']
        scores = cross_validate(model, 
                                X_train, y_train, 
                                cv=StratifiedKFold(n_splits=5), 
                                scoring=scoring)
        #print(scores)
        
        # Loss must be minimized
        best_score = scores['test_roc_auc'].mean()
        loss = 1 - best_score
    
        # Return negative loss (Hyperopt minimizes the objective function)
        return {
            'loss': loss, 
            'params': params,
            'accuracy': scores['test_accuracy'].mean(),
            'roc_auc': scores['test_roc_auc'].mean(),
            'status': STATUS_OK }
    
    # 4. Define the search space for the hyperparameters
    space = {
        'max_depth': scope.int(hp.quniform('max_depth', 2, 10, 1)),
        'min_samples_leaf':  scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
        'n_estimators' : hp.choice('n_estimators', [100, 120, 140])
    }

    # 5. Run Hyperopt to find the best hyperparameters
    trials = Trials()
    best = fmin(fn=objective, 
                space=space, 
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=trials)

    logger.info(f"Best hyperparameters found: {best}")

    # Return the evaluation metrics
    return {
        'best_params': best,
        'results': trials.results
    }


def train_xgb(
    X_train, y_train, #X_test, y_test,
    X_valid, y_valid,
    numerical_features, categorical_features,
    params_xgb
):   
    # Select features
    X_train = X_train[numerical_features + categorical_features]
    X_valid = X_valid[numerical_features + categorical_features]
    print(f"train: {X_train.shape}")
    print(f"valid: {X_valid.shape}")
    
    # Preprocessing
    transformers = []
    # Imputing categorical features)
    if categorical_features and len(categorical_features) > 0:
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
        )
        transformers.append(('cat', SimpleImputer(strategy='constant', fill_value=0), categorical_features))
    # Scaling numerical features
    transformers.append(('num', StandardScaler(), numerical_features))
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Create a pipeline with preprocessing and classification
    classifier = xgb.XGBClassifier(**params_xgb)

    # Train model
    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('classifier', classifier)
    ])
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_train_pred_proba = model.predict_proba(X_train)[::,1]
    y_valid_pred_proba = model.predict_proba(X_valid)[::,1]
    
    
    #calculate AUC of model
    train_acc = accuracy_score(y_train, y_train_pred_proba >= 0.5)
    test_acc = accuracy_score(y_valid, y_valid_pred_proba >= 0.5)
    test_auc = roc_auc_score(y_valid, y_valid_pred_proba)
    
    #print AUC score
    print(f"RF train acc: {train_acc:.3f}")
    print(f"RF valid acc: {test_acc:.3f}")
    print(f"RF valid auc: {test_auc:.3f}")

    return {
        'model': model,
        'metrics': {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_auc': test_auc,
        }
    }

# Function to optimize the classification model
def optimize_xgb(
    X_train, y_train,
    numerical_features, categorical_features, 
    max_evals=10
):
    
    # Select features
    X_train = X_train[numerical_features + categorical_features]
    n_features = len(numerical_features + categorical_features)
    logger.info(f"train: {X_train.shape}")
    
    # 3. Hyperparameter optimization using Hyperopt
    def objective(params):
        # Update the pipeline with the optimized hyperparameters
        logger.info(f"{params}")
        logger.info(f"train: {X_train.shape}")
        
        # Preprocessing
        transformers = []
        # Imputing categorical features)
        if categorical_features and len(categorical_features) > 0:
            transformers.append(('cat', SimpleImputer(strategy='constant', fill_value=0), categorical_features))
        # Scaling numerical features
        transformers.append(('num', StandardScaler(), numerical_features))
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Create a pipeline with preprocessing and classification
        classifier = xgb.XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            #subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            #min_child_weight=int(params['min_child_weight']),
            gamma=params['gamma'],
            #reg_alpha=params['reg_alpha'],
            #reg_lambda=params['reg_lambda'],
            objective='binary:logistic',  # Define this depending on tu objetivo
            #use_label_encoder=False,  # Disabling the deprecation warning
            eval_metric='logloss',  # Avoid warning
            random_state=42 
        )
    
        # Train model
        model = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('classifier', classifier)
        ])
        
        # Perform cross-validation to evaluate the model
        scoring=['accuracy',
                 'roc_auc']
        scores = cross_validate(model, 
                                 X_train, y_train, 
                                 cv=StratifiedKFold(n_splits=5), 
                                 scoring=scoring)
        #print(scores)
        
        # Loss must be minimized
        best_score = scores['test_roc_auc'].mean()
        loss = 1 - best_score
    
        # Return negative loss (Hyperopt minimizes the objective function)
        return {
            'loss': loss, 
            'params': params,
            'accuracy': scores['test_accuracy'].mean(),
            'roc_auc': scores['test_roc_auc'].mean(),
            'status': STATUS_OK }
    
    

    # 4. Define the search space for the hyperparameters
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 50)),  # Number of trees
        'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),  # Maximum depth of each tree
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),  # Learning rate
        #'subsample': hp.uniform('subsample', 0.5, 1.0),  # Fraction of data used for each tree
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.9),  # Fraction of features used for each tree
        #'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),  # Minimum sum of instance weight
        'gamma': hp.uniform('gamma', 0, 0.5),  # Minimum loss reduction for a split
        #'reg_alpha': hp.uniform('reg_alpha', 0, 1),  # L1 regularization term
        #'reg_lambda': hp.uniform('reg_lambda', 0, 1)  # L2 regularization terms
    }

    # 5. Run Hyperopt to find the best hyperparameters
    trials = Trials()
    best = fmin(fn=objective, 
                space=space, 
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=trials)

    logger.info("Best hyperparameters found:", best)
    #print("Best hyperparameters found:", trials.results)
    
    # Return the evaluation metrics
    return {
        'best_params': best,
        'results': trials.results
    }

def run_traininig(
    df_train, y_train, 
    df_valid, y_valid, 
    df_test, y_test,
    numerical_features, categorical_features, 
    trainer,
    model_params,
    model_name, 
    models_dir
):
    trained_model = trainer(
        df_train, y_train, 
        df_valid, y_valid, 
        numerical_features, categorical_features,
        model_params
    )
    metrics_train = trained_model['metrics']
    logger.info(f"[RUN-TRAINING] Train Metrics: {metrics_train}")
    model = trained_model['model']
    # Save Model
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    with open(model_path, 'wb') as file: 
        pickle.dump(model, file) 
    
    # Evaluation Base Random Forest
    metrics_eval = evaluate_model(
        model,
        df_test, y_test
    )
    #logger.info(f"[RUN-TRAINING] Metrics: {metrics_eval.shape}")
    metrics_eval_path = os.path.join(models_dir, f'{model_name}_eval.csv')
    metrics_eval.to_csv(metrics_eval_path, index=False)
    # Plot 
    metrics_eval_path = os.path.join(models_dir, f'{model_name}_eval_plot.png')
    plot_precision_recall(metrics_eval, output_file=metrics_eval_path)


def run_optimization(
    df_full_train, y_full_train,
    numerical_features, categorical_features, 
    optimizer,
    default_params,
    model_name,
    models_dir,
    max_evals=10
):
    # Optimize
    # Example of calling the optimization function
    optimize_results = optimizer(
        df_full_train, y_full_train, 
        numerical_features, categorical_features,
        max_evals=max_evals
    )
    opt_results_df = pd.DataFrame.from_dict(optimize_results['results'])
    opt_results_df = opt_results_df.sort_values(by='roc_auc', ascending=False)
    best_opt_params = opt_results_df.iloc[0].params
    best_params = {**default_params, **best_opt_params}
    logger.info(f"[OPIMIZATION] Best params: {best_params}")
    
    # Save best parameters
    best_params_file = os.path.join(models_dir, f'{model_name}_params.json')
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f)
    

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="Programa que recibe tres parámetros: fichero de datos, tipo de algoritmo y directorio de salida.")
    
    # Definir los argumentos obligatorios
    parser.add_argument('config', type=str, help="El fichero de datos (config) que se va a procesar")

    # Parsear los argumentos
    args = parser.parse_args()

    # Read raw data
    # Verificar si el fichero de datos existe
    if not os.path.isfile(args.config):
        print(f"Error: File {args.config} doesn't exist.")
        return
    # Read YAML file
    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)

    # Setting
    RAW_DATA_PATH = training_config["data_config"]["train_data"]["path"]
    TARGET = training_config["data_config"]["features"]["target"]
    RAW_DIR    = training_config["results"]["raw_dir"]
    PREPRO_DIR = training_config["results"]["prepro_dir"]
    MODELS_DIR = training_config["results"]["models_dir"]
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PREPRO_DIR, exist_ok=True)

    # Read raw data
    # Verificar si el fichero de datos existe
    if not os.path.isfile(RAW_DATA_PATH):
        print(f"Error: File {RAW_DATA_PATH} doesn't exist.")
        return
    data_filename = os.path.splitext(os.path.basename(RAW_DATA_PATH))[0]
    logger.info("filename: {data_filename}")
    raw_data = pd.read_csv(RAW_DATA_PATH)

    # Partition data
    seed = 42
    raw_full_train_data, raw_test_data = train_test_split(raw_data, test_size=0.2, random_state=seed)
    # Reset index
    raw_full_train_data = raw_full_train_data.reset_index(drop=True)
    raw_test_data = raw_test_data.reset_index(drop=True)
    # Save parition
    raw_full_train_path = os.path.join(RAW_DIR, f"{data_filename}_raw_train.csv")
    raw_full_train_data.to_csv(raw_full_train_path, index=False)
    raw_test_path = os.path.join(RAW_DIR, f"{data_filename}_raw_test.csv")
    raw_test_data.to_csv(raw_test_path, index=False)

    # Preprocessing
    logger.info(f"[data-raw] Columns: {raw_full_train_data.columns}")
    clean_full_train_data = run_data_wrangling(raw_full_train_data)
    clean_test_data = run_data_wrangling(raw_test_data)
    logger.info(f"[data-wrangling] Train Columns: {clean_full_train_data.columns}")
    logger.info(f"[data-wrangling] Test Columns : {clean_test_data.columns}")

    # Define columns
    TARGET = "blueWins"
    CATEGORICAL_FEATURES = clean_full_train_data.select_dtypes(include='category').columns.tolist()
    NUMERICAL_FEATURES = [col for col in clean_full_train_data.columns if col not in CATEGORICAL_FEATURES + [TARGET]]
    logger.info(f"Categorical Features: {CATEGORICAL_FEATURES}")
    logger.info(f"Numerical Features  : {NUMERICAL_FEATURES}")
    
    # Partition data
    clean_train_data, clean_valid_data = train_test_split(clean_full_train_data, test_size=0.25, random_state=seed)

    # Reset index
    clean_full_train_data = clean_full_train_data.reset_index(drop=True)
    clean_train_data = clean_train_data.reset_index(drop=True)
    clean_valid_data = clean_valid_data.reset_index(drop=True)
    clean_test_data = clean_test_data.reset_index(drop=True)
    # Save
    clean_full_train_data.to_csv(os.path.join(PREPRO_DIR, f'{data_filename}_full_train_clean.csv'))
    clean_train_data.to_csv(os.path.join(PREPRO_DIR, f'{data_filename}_train_clean.csv'))
    clean_valid_data.to_csv(os.path.join(PREPRO_DIR, f'{data_filename}_valid_clean.csv'))
    clean_test_data.to_csv(os.path.join(PREPRO_DIR, f'{data_filename}_test_clean.csv'))

    # Ys
    y_full_train = clean_full_train_data[TARGET].values
    y_train = clean_train_data[TARGET].values
    y_valid = clean_valid_data[TARGET].values
    y_test = clean_test_data[TARGET].values

    # Xs
    X_full_train = clean_full_train_data.drop(columns=[TARGET])
    X_train = clean_train_data.drop(columns=[TARGET])
    X_valid = clean_valid_data.drop(columns=[TARGET])
    X_test = clean_test_data.drop(columns=[TARGET])

    # Feature Selection
    NUMERICAL_FEATURES_FS, CATEGORICAL_FEATURES_FS = run_feature_selection(
        X_train, y_train,
        X_valid, y_valid, 
        NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    )
    logger.info(f"[feature-selection] Categorical Features: {CATEGORICAL_FEATURES_FS}")
    logger.info(f"[feature-selection] Numerical Features  : {NUMERICAL_FEATURES_FS}")
    data_features = {
        "numerical": NUMERICAL_FEATURES_FS, 
        "categorical": CATEGORICAL_FEATURES_FS,
        "target": TARGET
    }
    data_features_path = os.path.join(MODELS_DIR, "features.json")
    with open(data_features_path, 'w') as f:
        json.dump(data_features, f)
    

    # Train Base Random Forest
    base_model_name_rf = 'base_rf'
    base_param_rf = {
        'n_estimators':10, 
        'min_samples_leaf': 10,
        'random_state':42
    }
    run_traininig(
        X_train, y_train, 
        X_valid, y_valid, 
        X_test, y_test,
        NUMERICAL_FEATURES_FS, CATEGORICAL_FEATURES_FS, 
        train_rf,
        base_param_rf,
        base_model_name_rf, 
        MODELS_DIR
    )

    # Optimize Random Forest
    best_model_name_rf = 'best_rf'
    best_default_params_rf = {'random_state': 42}
    run_optimization(
        X_full_train, y_full_train,
        NUMERICAL_FEATURES_FS, CATEGORICAL_FEATURES_FS, 
        optimize_rf,
        best_default_params_rf,
        best_model_name_rf,
        MODELS_DIR,
        max_evals=20
    )
    # Train Best Random Forest
    base_param_rf_file = os.path.join(MODELS_DIR, f'{best_model_name_rf}_params.json')
    with open(base_param_rf_file, 'r') as f:
        base_param_rf = json.load(f)
    run_traininig(
        X_train, y_train, 
        X_valid, y_valid, 
        X_test, y_test,
        NUMERICAL_FEATURES_FS, CATEGORICAL_FEATURES_FS, 
        train_rf,
        base_param_rf,
        best_model_name_rf, 
        MODELS_DIR
    )

    # Train Base XGB
    base_model_name_xgb = 'base_xgb'
    base_params_xgb = {
        'tree_method': "hist", 
        'enable_categorical': True,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic'
    }
    run_traininig(
        X_train, y_train, 
        X_valid, y_valid, 
        X_test, y_test,
        NUMERICAL_FEATURES_FS, CATEGORICAL_FEATURES_FS, 
        train_xgb,
        base_params_xgb,
        base_model_name_xgb, 
        MODELS_DIR
    )

    # Optimize XGB
    best_model_name_xgb = 'best_xgb'
    best_default_params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    }
    run_optimization(
        X_full_train, y_full_train,
        NUMERICAL_FEATURES_FS, CATEGORICAL_FEATURES_FS, 
        optimize_xgb,
        best_default_params_xgb,
        best_model_name_xgb,
        MODELS_DIR,
        max_evals=10
    )
    # Train Best Random Forest
    base_param_rf_file = os.path.join(MODELS_DIR, f'{best_model_name_xgb}_params.json')
    with open(base_param_rf_file, 'r') as f:
        base_param_xgb = json.load(f)
    run_traininig(
        X_train, y_train, 
        X_valid, y_valid, 
        X_test, y_test,
        NUMERICAL_FEATURES_FS, CATEGORICAL_FEATURES_FS, 
        train_xgb,
        base_param_xgb,
        best_model_name_xgb, 
        MODELS_DIR
    )

if __name__ == "__main__":
    main()

