import os
import yaml
import argparse
# Files
import json
import pickle 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# Logging
import logging

# Crear un logger
logger = logging.getLogger()

# Configuración para el archivo
os.makedirs("logs", exist_ok=True)
log_file_path = os.path.join("logs",'predict-logs.log')
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


def predict(model, X):
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[::,1]

    return y_pred, y_pred_prob


def run_predict(
        model,
        X_test, 
        y_test,
        run_exp_name,
        output_dir
):    
    # Predict
    y_test_pred, y_test_pred_proba = predict(model, X_test)
    
    # Save prediction
    df_predict = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_pred_proba': y_test_pred_proba
    })
    df_predict.to_csv(os.path.join(output_dir, f'{run_exp_name}_predictions.csv'), index=False)

    # Metrics: ACC, AUC
    test_acc = accuracy_score(y_test, y_test_pred_proba >= 0.5)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_metrics_df = pd.DataFrame([{'test_acc': test_acc, 'test_auc': test_auc}])
    test_metrics_df.to_csv(os.path.join(output_dir, f'{run_exp_name}_metrics.csv'), index=False)

    # Metrics: Precision Recall F1
    precision_recall_f1_df = calculate_precision_recall_f1(y_test, y_test_pred_proba)
    precision_recall_f1_df.to_csv(os.path.join(output_dir, f'{run_exp_name}_precision_recall_f1.csv'), index=False)

    # Plot Prediction
    plot_precision_recall(
        precision_recall_f1_df, 
        output_file=os.path.join(output_dir, f'{run_exp_name}_precision_recall_plot.png')
    )

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="Programa que recibe tres parámetros: fichero de datos, tipo de algoritmo y directorio de salida.")
    
    # Definir los argumentos obligatorios
    parser.add_argument('config', type=str, help="El fichero de datos (raw_data) que se va a procesar")

    # Parsear los argumentos
    args = parser.parse_args()

    # Read raw data
    # Verificar si el fichero de datos existe
    if not os.path.isfile(args.config):
        print(f"Error: File {args.config} doesn't exist.")
        return

    # Read YAML file
    with open(args.config, 'r') as f:
        prediction_config = yaml.safe_load(f)

    # Setting
    TEST_DATA = prediction_config["data_config"]["test_data"]
    OUTPUT_DIR = prediction_config["prediction_dir"]
    MODELS_DIR = prediction_config["models_dir"]
    NUMERICAL_FEATURES = prediction_config["data_config"]["features"]["numerical"]
    CATEGORICAL_FEATURES = prediction_config["data_config"]["features"]["categorical"]
    TARGET = prediction_config["data_config"]["features"]["target"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_filename in os.listdir(MODELS_DIR):
        if not model_filename.endswith('.pkl'):
            #logger.info(f"Skip {filename}")
            continue
        
        # Model name
        model_name = os.path.splitext(os.path.basename(model_filename))[0]

        # Load model
        model_path = os.path.join(MODELS_DIR, model_filename)
        logger.info(f'model_path: {model_path}')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        for data_conf in TEST_DATA:
            logger.info(f'model_path: {data_conf}')
            test_kind = data_conf['kind']
            test_path = data_conf['path']
            # Check if exist file
            if not os.path.exists(test_path):
                logger.info(f"File {test_path} doesn't exist")
                continue
            
            # Read data file
            test_data = pd.read_csv(test_path)
            if test_kind == 'raw':
                test_data = run_data_wrangling(test_data)
            
            y_test = test_data[TARGET].values
            X_test = test_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
            run_exp_name = f'{model_name}_{test_kind}'
            exp_results_path = os.path.join(OUTPUT_DIR, run_exp_name)
            os.makedirs(exp_results_path, exist_ok=True)
            # Run preductions
            run_predict(
                model,
                X_test, y_test,
                run_exp_name,
                output_dir=exp_results_path
            )

if __name__ == "__main__":
    main()

