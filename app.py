from flask import Flask, request, jsonify
import os
import pickle
import json
import yaml
import numpy as np
import pandas as pd
# Logging
import logging

# Crear un logger
logger = logging.getLogger()

# Configuración para el archivo
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(os.path.join("logs", 'app-logs.log'))
file_handler.setLevel(logging.INFO)  # Puedes ajustar el nivel aquí (INFO, DEBUG, etc.)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# Configuración para la consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Igualmente, puedes ajustar el nivel de consola
console_handler.setFormatter(file_formatter)


# Inicializar la aplicación Flask
app = Flask(__name__)

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

def make_setting(config_path):
    # Read YAML file
    with open(config_path, 'r') as f:
        APP_CONFIG = yaml.safe_load(f)

    # Read the JSON file
    logger.info(f"app_config: {APP_CONFIG}")
    features_path = APP_CONFIG["features"]["path"]
    with open(features_path, 'r') as file:
        data_features = json.load(file)

    # Setting features
    global NUMERICAL_FEATURES
    global CATEGORICAL_FEATURES
    global TARGET
    NUMERICAL_FEATURES = data_features["numerical"]
    CATEGORICAL_FEATURES = data_features["categorical"]
    TARGET = data_features["target"]

    # Load models
    global MODEL_MAP
    MODEL_MAP = {}
    models = APP_CONFIG["models"]
    for model_conf in models:
        model_name = model_conf["name"]
        model_path = model_conf["path"]
        if not os.path.exists(model_path):
            logger.info(f"File {model_path} doesn't exist")
            continue
        # Cargar los clasificadores desde los archivos pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        MODEL_MAP[model_name] = model

CONFIG_PATH = os.path.join("config", "app_config.yaml")
make_setting(CONFIG_PATH)

# Definir el endpoint de predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del request JSON
    raw_data = request.get_json()
    print(f"raw_data: {raw_data}")
    logger.info(f"raw_data: {raw_data}")
    if isinstance(raw_data, dict):
        data = [raw_data]
    else:
        data = raw_data

    data_df = pd.DataFrame(data)

    clean_data = run_data_wrangling(data_df)
    clean_data = clean_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    '''
    # Asegurarse que todos los campos necesarios estén en el request
    required_fields = ['blueGoldDiff', 'blueExperienceDiff', 'blueKDADiff', 
                       'blueDragons', 'blueCSDiff', 'blueJGCSDiff', 'redDragons', 'blueFirstBlood']

    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Faltan algunos campos en los datos proporcionados'}), 400
    

    # Extraer las características del cuerpo de la petición
    features = np.array([[data[field] for field in required_fields]])
    '''

    # Realizar la predicción con ambos modelos
    predictions = {}
    for model_name, model in MODEL_MAP.items():
        preds = model.predict(clean_data)
        predictions[model_name] = preds

    # Opcionalmente, puedes retornar las probabilidades en lugar de las predicciones
    # xgb_probability = xgb_model.predict_proba(features)[:, 1]  # Probabilidad de clase 1 (blueWins = 1)
    # rf_probability = rf_model.predict_proba(features)[:, 1]

    # Devolver la respuesta como JSON con las predicciones de ambos modelos
    '''
    return jsonify({
        'xgb_prediction': int(xgb_prediction[0]),
        'rf_prediction': int(rf_prediction[0]),
        # 'xgb_probability': float(xgb_probability[0]),
        # 'rf_probability': float(rf_probability[0]),
    })
    '''
    results = pd.DataFrame(predictions).to_dict('records')
    print(f"results: {results}")
    return jsonify(results)

# Correr la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
