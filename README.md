# Project: League of Legends Diamond Ranked Games (10 min)

<p align="center">
  <img src="images/banner.jpg">
</p>


## Dataset

In this section, the League of Legends dataset selected for this work is described. To better understand the content of the dataset, a brief introduction to how League of Legends works is provided. Finally, the structure of this dataset is explained.

### Introduction

[League of Legends](https://www.leagueoflegends.com/es-es/) is a multiplayer online battle arena ([MOBA](https://en.wikipedia.org/wiki/Multiplayer_online_battle_arena)) video game developed by Riot Games. League of Legends is one of the most popular online games. In 2020 alone, League of Legends had over 110 million active players, which shows that even though the game is over 10 years old, it is still quite popular; it also shows that new players are creating accounts and joining.

In this game, there are two teams, Red and Blue, each consisting of 5 players. The goal of each team is to destroy the other team's base of operations (Nexus). The team that does so first wins the game. League of Legends matches can be quite long since as long as both teams' bases remain standing, the game continues. The average game length according to Riot Games is around [30 minutes](https://www.leagueofgraphs.com/stats/game-durations).

Before the game starts, each player in the game must choose a unique champion. As of November 2, 2022, there are [162 champions](https://leagueoflegends.fandom.com/wiki/Champion) in the game to choose from, but that number increases frequently as Riot constantly introduces new champions to the game. The game gets complex quickly as beyond the 10 champions that are in the game (5 on your team and 5 on the other), there are many systems to take into account, such as the economic system for purchasing items, the experience system to allow your champion access to more powerful abilities, and neutral objectives that grant powerful effects.

####  Ranked Games

League of Legends still uses a form of ranking system inspired by the Elo system called the League system, matching players with similar skill levels to play with and against each other. It consists of **nine tiers** that indicate the **skill level of the players**. Players within each division are ranked using a points system called League Points (LP). Every time you play, you will lose or gain LP depending on whether you win or lose the game; the more LP you gain, the higher your Elo ranking can rise.

Below is a representation of all the tiers currently in League of Legends, starting from **Iron** (the lowest) on the left, followed by Bronze, **Silver**, **Gold**, **Platinum**, **Diamond**, **Master**, **Grandmaster**, and finally **Challenger** (the highest).

To see the full ranking distribution, you can visit the [rank distribution](https://www.leagueofgraphs.com/rankings/rank-distribution) link. It is considered that Diamond players are in the top 2.4% of players, meaning their understanding of the game is deep, and they generally make high-impact decisions that tend to pay off. Therefore, tracking their behavior and seeing what leads to victories in their games is an accurate way to predict who will win a game of League of Legends.


#### Glossary

- **Solo Queue**. The [**Solo Queue**](https://bloygo.yoigo.com/glosario/definicion-soloq/) or **SoloQ** is a game mode where gamers can **improve their skills by playing solo**. In this way, the SoloQ Challenge is a way to exploit a style of play that would otherwise remain in the shadows. It is typical of **League of Legends** because it is a sort of **purgatory where the player practices alone** in order to improve their skills.

- **Elo**. [**Elo**](https://leagueoflegends.fandom.com/wiki/Elo_rating_system#:~:text=A%20person%20with%20a%20higher,in%20relation%20to%20other%20players.) is essentially the ranking of a player. It is determined by the win/loss ratio and how many games a player has played. In a **normal Queue**, the Elo is hidden from the player and others, but the matchmaking system still uses it to decide the player's opponents. In ranked games, the player's Elo is visible to everyone, with **Iron being the lowest and Challenger the highest**.

- **Nexus**. The [**Nexus**](https://leagueoflegends.fandom.com/wiki/Nexus) is a structure that serves as the primary objective in all game modes in League of Legends. *The team that destroys the enemy's Nexus wins the match.*

- **Ward**. A Ward is a **guardian** in **LoL**. Wards allow you to see beyond the fog of war, which is crucial for, for example, **deciding when and if you should attack, and when is the best time to do so**. If the game is not going well, it’s best to place wards in our jungle, while if we are doing well, we can risk putting them in the enemy jungle to **see what our opponent is doing**.

- **Minion**. [**Minions**](https://leagueoflegends.fandom.com/wiki/Minion_(League_of_Legends)) are units that make up the main force sent by the Nexus. They are periodically generated from their Nexus and move along a lane toward the enemy Nexus, automatically engaging any enemy units or structures they encounter. They are controlled by artificial intelligence and only use basic attacks.

- **CS**. The minion death count is a recorded score, commonly known as [**Creep Score (CS)**](https://leagueoflegends.fandom.com/wiki/Farming).

- **Buff**. A [**buff**](https://leagueoflegends.fandom.com/wiki/Buff) (benefit) is any status effect granted to a champion or minion that provides a boost to their performance. The opposite is called a **debuff**. Buffs can enhance almost any attribute of a character, such as health, mana, attack damage, and ability power, but in some cases, they may provide more than just statistical changes.

- **First Blood**. [**First Blood**](https://www.pinnacle.com/en/esports-hub/betting-articles/league-of-legends/betting-on-first-blood/sgajzgnuz8lgxujv) (FB) refers to a League of Legends (LoL) team that achieves the first player kill of an enemy during a game. Getting FB on an opponent is crucial due to the impact it has in the early stages of the game.

- **Champions**. [**Champions**](https://leagueoflegends.fandom.com/wiki/Champion) are the characters controlled by the player in League of Legends. Each champion has unique abilities and attributes.

- **Turrets**. [**Turrets**](https://leagueoflegends.fandom.com/wiki/Turret) (also called towers) are strong fortifications that attack enemy units in sight. Turrets are a central component of League of Legends. They deal damage to enemies and provide vision to their team, allowing them to better control the battlefield. Turrets target one unit at a time and deal significant damage. Teams must destroy enemy turrets to push their assault into enemy territory.


### Description

To do this work we have selected the dataset [diamond ranking game](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min) from Kaggle. This dataset contains the first 10 minutes, statistics of about 10k ranked games (in SOLO QUEUE mode) of a high ELO (from DIAMOND I to MASTER). The players are of approximately the same level.

Within this dataset there are two features of interest (gameId and blueWins) and we will explain them before.

The feature **gameId** represents a single game. The gameId field can be useful to get more attributes from the Riot API.

The feature **blueWins** is the target value (the value we are trying to predict). A value of 1 means that the blue team has won. 0 otherwise.

There are 19 traits per team (38 total) collected after 10 minutes in-game. This includes kills, deaths, gold, experience, level, etc. It's up to you to do some feature engineering to get more information. Variables belonging to the blue team are prefixed with **blue**, while variables for the red team are prefixed with **red**.

The following briefly describes these traits:

- **blueFirstBlood, redFirstBlood**. Represents which team made the first kill. The value 1 represents that a team made the first kill, and 0 otherwise.

- **blueKills, redKills**. Represents the kills made by each team. A direct metric of how good the team is.

- **blueDeath, redDeath**. These are inverted variables of the above.

- **blueAssits, redAssits**. Represents the attacks by players from each team, which did not lead to the final deal. In a way, these stats represent team collaboration.

- **blueAvgLevel, redAvgLevel** - These represent the average level of champions per team.

- **blueTotalExperience, redTotalExperience, blueTotalGold, redTotalGold** - These represent the experience and gold stats earned by each team.

- **blueDragons, blueHeralds, redDragons, redHeralds** - These represent how many dragons and heralds a team killed.

- **blueEliteMonsters, redEliteMonsters** - These represent the number of high priority neutral monsters a team killed.

- **blueTotalMinionsKilled, redTotalMinionsKilled** - These represent the number of monsters killed by a team.

- **blueTotalJungleMinionsKilled, redTotalJungleMinionsKilled** - These represent the number of monsters killed by a team. Represents the number of jungle monsters killed by a team.

- **blueWardsPlaced, blueWardsDestroyed, redWardsPlaced, redWardsDestroyed** - **Wards** are one of the deployable units for different benefits. So the number of placements or destructions represents an active or dominant game in the first 10 minutes.

- **blueTowersDestroyed, redTowersDestroyed** - Represents the number of opponent turrets destroyed by each team. Also a useful index of aggressive game development.

- **blueGoldDiff, redGoldDiff** - Represents the difference in the amount of gold between teams. These metrics are calculated by the differences in the total amount of gold of the teams.

- **blueExperienceDiff, redExperienceDiff** - Represents the difference in the amount of experience between teams. These metrics are calculated by the differences in the total amount of experience of the teams.

- **blueCSPerMin, redCSPerMin**. Represents the number of minion kills (CS) per minute for a team.

- **blueGoldPerMin, redGoldPerMin**. Represents the amount of gold earned per minute for a team.

Within these characteristics there are some that are the result of additions, comparisons or additions of other variables. These variables can represent redundant information that must be taken into consideration.

In the case of aggregate variables, these are those that result from a calculation, such as an average. These variables can be identified by having the suffix **PerMin** or containing **Avg** in their names.

Other variables are of a comparative type that are the result of the difference between the totals (of something) of the teams. These types of variables are identified by having the suffix **Diff** in their name.

Finally, in the group of additive variables we only have the elite monster counts that are the sum of dragons and heralds.


The file containing the dataset is stored in [high_diamond_ranked_10min.csv](data/raw/high_diamond_ranked_10min.csv).

## Technologies

- Python 3.10.12
- [Flask 3.0.3](https://flask.palletsprojects.com/en/stable/) is a lightweight and flexible web framework for Python that allows developers to build web applications quickly and with minimal overhead. It provides essential tools for routing, templating, and handling HTTP requests, while allowing for easy extensibility through a wide range of plugins and extensions.
- [Hyperopt 0.2.7](https://hyperopt.github.io/hyperopt/) is a Python library used for optimizing machine learning models through hyperparameter tuning. It offers efficient algorithms like random search, grid search, and Bayesian optimization, helping to automatically find the best hyperparameters for a given model.
- Docker and Docker Compose for containerization
- [XGBoost 2.1.2](https://xgboost.readthedocs.io/en/stable/) is a powerful and efficient machine learning library that implements gradient boosting algorithms for supervised learning tasks, particularly in classification and regression problems. It is known for its high performance, scalability, and accuracy, making it a popular choice for data science competitions and real-world applications.
- [Scikit-learn 1.5.2](https://scikit-learn.org/stable/) is a popular Python library for building and evaluating machine learning models, providing efficient tools for classification, regression, clustering, and dimensionality reduction. It offers a simple and consistent interface, along with a wide range of algorithms and data preprocessing techniques, making it easy to integrate into data analysis and artificial intelligence projects.
- [Pipenv](https://pipenv.pypa.io/en/latest/) is a tool for managing dependencies in Python projects, combining the functionalities of pip and virtualenv.
- [Jupyter](https://jupyter.org/) is an open-source web application that allows users to create and share documents containing live code, equations, visualizations, and narrative text. It supports various programming languages, including Python, R, and Julia, making it widely used for data science, machine learning, and academic research. Its interactive environment enables real-time code execution, making it an essential tool for exploratory data analysis and visualization.



## Architecture

The application code is composed of the following components:

- [`app.py`](app.py) - Module with Flask application.
- [`train.py`](train.py) - Module for preprocessing, feature selection and training with XGBoost and Random Forest models.
- [`predict.py`](predict.py) - Module to obtain predictions with the test subset for the XGBoost and Random Forest models.
- [`test_app.py`](test_app.py) - Module to test the Flask application from the test subset.
- [`Dockerfile`](Dockerfile) - Dockerfile to build an image for the Flask application that returns predictions for the XGBoost and Random Forest models.
- [`docker-compose`](docker-compose.yaml) - Docker compose serving the Flask application on port 5000.


The configuration for the application is in the [`config/`](config/) folder:

- [`app_config.yaml`](config/app_config.yaml) - Flask Application configuration data
- [`train_config.yaml`](config/train_config.yaml)  - Configuration data for Training.
- [`predict_config.yaml`](config/predict_config.yaml)  - Configuration data for Testing.
- [`test_app_config.yaml`](config/test_app_config.yaml)  - Configuration data for App Flask Testing.

Log files are stored in the [logs](logs) directory.

## Preparation

For dependency management, we use pipenv, so you need to install it:

```bash
pip install pipenv
```

Once installed, you can install the app dependencies:

```bash
pipenv install --dev
```

## Running the application


### Running with Docker-Compose

The easiest way to run the application is with `docker-compose`. 

First we build the docker image.

```bash
docker-compose build
```

Then, we boot the image with the following command:

```bash
docker-compose up -d
```

### Running with Docker (without compose)

Sometimes you might want to run the application in
Docker without Docker Compose, e.g., for debugging purposes.

First, prepare the environment by running Docker Compose
as in the previous section.

Next, build the image:

```bash
docker build -t ml-zoomcamp-midterm-lol:3.10.12-slim . 
```
Run it:

```bash
docker run -it --rm \
    -p 5000:5000 \
    ml-zoomcamp-midterm-lol:3.10.12-slim
```

### Running Flask 

We can also start the Flask application as a Python application with the following command.

```bash
pipenv shell

python app.py 
```

## Experiments

For experiments, we use Jupyter notebooks.
They are in the [`notebooks`](notebooks/) folder.

To start Jupyter, run:

```bash
pipenv shell

cd notebooks

jupyter notebook
```

We have the following notebooks:

- [`notebook.ipynb`](notebooks/notebook.ipynb): Notebook for training, evaluation and optimization with the XGBoost and Random Forest models.

## Training
The [train.py](train.py) module contains the logic to perform the preprocessing of the raw data, feature selection, training and optimization of the XGBoost and Random Forest models. This module requires the configuration file [train_config.yaml](config/train_config.yaml) with the following parameters:

- `data_config.features.target`: Features to predict in the dataset.
- `data_config.train_data.path`: File with the dataset.
- `results.raw_dir`: Directory where the unpreprocessed subsets are stored.
- `results:prepro_dir`: Directory where the preprocessed subsets are stored.
- `results.models_dir`: Directory where the models are stored.

The following describes the configuration file [train_config.yaml](config/train_config.yaml)

```yaml
data_config:
  features:
    target: 'blueWins'
  train_data:
      path: data/raw/high_diamond_ranked_10min.csv

results:
  raw_dir: "data/raw"
  prepro_dir: "data/prepro"
  models_dir: "models"
```
During the training phase, models are generated for `XGBoost` and `Random Forest`. For `Random Forest`, a base model and an optimized model are generated, the same applies to `XGBoost`. These generated models are saved in the `models` directory and their content is shown below.

```bash
ls -lh models/
total 1,9M
-rw-rw-r-- 1 aztleclan aztleclan 6,2K nov 25 17:17 base_rf_eval.csv
-rw-rw-r-- 1 aztleclan aztleclan 143K nov 25 17:17 base_rf_eval_plot.png
-rw-rw-r-- 1 aztleclan aztleclan 422K nov 25 17:16 base_rf.pkl
-rw-rw-r-- 1 aztleclan aztleclan 6,2K nov 25 17:17 base_xgb_eval.csv
-rw-rw-r-- 1 aztleclan aztleclan 148K nov 25 17:17 base_xgb_eval_plot.png
-rw-rw-r-- 1 aztleclan aztleclan 270K nov 25 17:17 base_xgb.pkl
-rw-rw-r-- 1 aztleclan aztleclan 5,9K nov 25 17:17 best_rf_eval.csv
-rw-rw-r-- 1 aztleclan aztleclan 145K nov 25 17:17 best_rf_eval_plot.png
-rw-rw-r-- 1 aztleclan aztleclan   80 nov 25 13:37 best_rf_param.json
-rw-rw-r-- 1 aztleclan aztleclan   80 nov 25 17:17 best_rf_params.json
-rw-rw-r-- 1 aztleclan aztleclan 508K nov 25 17:17 best_rf.pkl
-rw-rw-r-- 1 aztleclan aztleclan 6,1K nov 25 17:17 best_xgb_eval.csv
-rw-rw-r-- 1 aztleclan aztleclan 148K nov 25 17:17 best_xgb_eval_plot.png
-rw-rw-r-- 1 aztleclan aztleclan  348 nov 25 17:17 best_xgb_params.json
-rw-rw-r-- 1 aztleclan aztleclan  77K nov 25 17:17 best_xgb.pkl
-rw-rw-r-- 1 aztleclan aztleclan  184 nov 25 17:30 features.json
```

- [base_rf.pkl](models/base_rf.pkl): Base model trained with Random Forest.
- [best_rf.pkl](models/best_rf.pkl): Model optimized with Random Forest.
- [best_rf_params.json](models/best_rf_params.json): Best parameters for model optimized with Random Forest.
- [base_xgb.pkl](models/base_xgb.pkl): Base model trained with XGBoost.
- [best_xgb.pkl](models/best_xgb.pkl): Model optimized with XGBoost.
- [best_xgb_params.json](models/best_xgb_params.json): Best parameters for model optimized with XGBoost.
- [features.json](models/features.json): It contains the numerical, categorical and target features extracted in the feature selection phase.

The content of the features.json file is shown below:

```json
{
  "numerical": [
    "blueGoldDiff",
    "blueExperienceDiff",
    "blueKDADiff",
    "blueCSDiff",
    "blueDragons",
    "blueJGCSDiff",
    "redDragons"
  ],
  "categorical": [
    "blueFirstBlood"
  ],
  "target": "blueWins"
}
```

## Prediction

The [predict.py](predict.py) module contains the logic to obtain the predictions of the test subset. This module requires the configuration file [predict_config.yaml](config/predict_config.yaml) with the following parameters:

- `data_config.features.path`: File with the types of features of the dataset.
- `test_data.kind`: Type of test subset. If **raw** is specified, preprocessing must be performed before making the predictions. If it has the value **prepro** the test subset is already preprocessed.
- `test_data.path`: File containing the test subset.
- `models_dir`: Directory where the XBGBoost and Random Forest models are stored.
- `prediction_dir`: Directory where the predictions obtained from the test subset are stored.

The following describes the configuration file [predict_config.yaml](config/predict_config.yaml)

```yaml
data_config:
  features:
    path: models/features.json
  test_data:
    - kind: raw
      path: data/raw/high_diamond_ranked_10min_raw_test.csv
    - kind: prepro
      path: data/prepro/high_diamond_ranked_10min_test_clean.csv


models_dir: "models"
prediction_dir: "output/predict"
```

To run this module, execute the following command:

```bash
pipenv shell

python predict.py config/predict_config.yaml
```
The prediction results for the different test subsets and XGBoost and Random Forest models are stored in the `output/predict` directory. The contents of the directory are shown below.

```bash
ls -lh output/predict/

total 32K
drwxrwxr-x 2 aztleclan aztleclan 4,0K nov 25 19:06 base_rf_prepro
drwxrwxr-x 2 aztleclan aztleclan 4,0K nov 25 19:06 base_rf_raw
drwxrwxr-x 2 aztleclan aztleclan 4,0K nov 25 19:06 base_xgb_prepro
drwxrwxr-x 2 aztleclan aztleclan 4,0K nov 25 19:06 base_xgb_raw
drwxrwxr-x 2 aztleclan aztleclan 4,0K nov 25 19:06 best_rf_prepro
drwxrwxr-x 2 aztleclan aztleclan 4,0K nov 25 19:06 best_rf_raw
drwxrwxr-x 2 aztleclan aztleclan 4,0K nov 25 19:06 best_xgb_prepro
drwxrwxr-x 2 aztleclan aztleclan 4,0K nov 25 19:06 best_xgb_raw
```
While in the `output/predict/best_rf_prepro` directory the predictions of the Random Forest model for the preprocessed test subset are shown.

```bash
ls -lh output/predict/best_rf_prepro/

total 208K
-rw-rw-r-- 1 aztleclan aztleclan   55 nov 25 22:28 best_rf_prepro_metrics.csv
-rw-rw-r-- 1 aztleclan aztleclan 5,9K nov 25 22:28 best_rf_prepro_precision_recall_f1.csv
-rw-rw-r-- 1 aztleclan aztleclan 145K nov 25 22:28 best_rf_prepro_precision_recall_plot.png
-rw-rw-r-- 1 aztleclan aztleclan  45K nov 25 22:28 best_rf_prepro_predictions.csv
```
As you can see, four files have been created, which are described below.

- `best_rf_prepro_metrics.csv`. Average of the metrics of the test subset.
- `best_rf_prepro_precision_recall_f1.csv`. Precision, recall and f1 metrics for different thresholds.
- `best_rf_prepro_precision_recall_plot.png`. Graph with the precision and recall metrics.
- `best_rf_prepro_predictions.csv`. Predictions obtained from the test subset.s


## Test Flasks

The [app.py](app.py) module contains the logic for testing the Flask application. This module needs the configuration file [test_app_config.yaml](config/test_app_config.yaml) with the following parameters:

- `data_config.features.path`: File with the dataset features.
- `data_config.test_data.path`: File with the test subset in csv format.
- `endpoints.predict`: URL of the Flask application's prediction method.
- `outputs.path`: Directory where the predictions returned by the Flask application are saved.

The following describes the configuration file [test_app_config.yaml](config/test_app_config.yaml)

```yaml
data_config:
  features:
    path: models/features.json
  test_data:
      path: data/raw/high_diamond_ranked_10min_raw_test.csv

endpoints:
  predict: "http://127.0.0.1:5000/predict"

outputs:
  path: output/test_app
```

To run this module, execute the following command:

```bash
pipenv shell

python test_app.py config/test_app_config.yaml 
```
For example, for the following entry data in json format:

```json
[
    {
        "gameId": 4519205334,
        "blueWardsPlaced": 14,
        "blueWardsDestroyed": 2,
        "blueFirstBlood": 0,
        "blueKills": 4,
        "blueDeaths": 8,
        "blueAssists": 3,
        "blueEliteMonsters": 2,
        "blueDragons": 1,
        "blueHeralds": 1,
        "blueTowersDestroyed": 0,
        "blueTotalGold": 14885,
        "blueAvgLevel": 6.8,
        "blueTotalExperience": 17810,
        "blueTotalMinionsKilled": 192,
        "blueTotalJungleMinionsKilled": 59,
        "blueGoldDiff": -4790,
        "blueExperienceDiff": -2126,
        "blueCSPerMin": 19.2,
        "blueGoldPerMin": 1488.5,
        "redWardsPlaced": 89,
        "redWardsDestroyed": 2,
        "redFirstBlood": 1,
        "redKills": 8,
        "redDeaths": 4,
        "redAssists": 5,
        "redEliteMonsters": 0,
        "redDragons": 0,
        "redHeralds": 0,
        "redTowersDestroyed": 1,
        "redTotalGold": 19675,
        "redAvgLevel": 7.4,
        "redTotalExperience": 19936,
        "redTotalMinionsKilled": 232,
        "redTotalJungleMinionsKilled": 67,
        "redGoldDiff": 4790,
        "redExperienceDiff": 2126,
        "redCSPerMin": 23.2,
        "redGoldPerMin": 1967.5
    },
    {
        "gameId": 4501108199,
        "blueWardsPlaced": 15,
        "blueWardsDestroyed": 1,
        "blueFirstBlood": 1,
        "blueKills": 10,
        "blueDeaths": 5,
        "blueAssists": 10,
        "blueEliteMonsters": 0,
        "blueDragons": 0,
        "blueHeralds": 0,
        "blueTowersDestroyed": 0,
        "blueTotalGold": 17878,
        "blueAvgLevel": 7.0,
        "blueTotalExperience": 18248,
        "blueTotalMinionsKilled": 214,
        "blueTotalJungleMinionsKilled": 52,
        "blueGoldDiff": 2301,
        "blueExperienceDiff": 603,
        "blueCSPerMin": 21.4,
        "blueGoldPerMin": 1787.8,
        "redWardsPlaced": 37,
        "redWardsDestroyed": 1,
        "redFirstBlood": 0,
        "redKills": 5,
        "redDeaths": 10,
        "redAssists": 5,
        "redEliteMonsters": 0,
        "redDragons": 0,
        "redHeralds": 0,
        "redTowersDestroyed": 0,
        "redTotalGold": 15577,
        "redAvgLevel": 6.8,
        "redTotalExperience": 17645,
        "redTotalMinionsKilled": 203,
        "redTotalJungleMinionsKilled": 44,
        "redGoldDiff": -2301,
        "redExperienceDiff": -603,
        "redCSPerMin": 20.3,
        "redGoldPerMin": 1557.7
    }
]
```

The Flask application returns the following results:

```json
[
    {
        "base_rf_cls": 0,
        "base_xgb_cls": 0,
        "best_rf_cls": 0,
        "best_xgb_cls": 0
    },
    {
        "base_rf_cls": 1,
        "base_xgb_cls": 1,
        "best_rf_cls": 1,
        "best_xgb_cls": 1
    }
]
```

It can be observed that the result has four predictions that correspond to four models:

- `base_rf_cls`: Prediction result of the base Random Forest model.
- `base_xgb_cls`: Prediction result of the base XGBoost model.
- `best_rf_cls`: Prediction result of the optimized Random Forest model.
- `best_xgb_cls`: Prediction result of the optimized XGBoost model.
