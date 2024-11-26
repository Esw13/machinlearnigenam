import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import tensorflow as tf
import kerastuner as kt

# Étape 1 : Charger et Préparer les Données
def load_and_preprocess_data(file_path, feature_column='Close', train_ratio=0.8, window_size=60):
    data = pd.read_csv(file_path)
    data = data[[feature_column]]

    # Normalisation des données
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Séparation entraînement/test
    train_size = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - window_size:]

    return data, train_data, test_data, scaler

# Créer des séquences pour LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

# Étape 2 : Construire un modèle LSTM avec optimisation des hyperparamètres
def build_lstm_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int('units', min_value=50, max_value=200, step=50),
            return_sequences=True,
            input_shape=(window_size, 1),
        )
    )
    model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(
        LSTM(
            units=hp.Int('units2', min_value=50, max_value=200, step=50),
            return_sequences=False,
        )
    )
    model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', min_value=25, max_value=100, step=25)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Optimiser le modèle avec Keras Tuner
def optimize_model(X_train, y_train):
    tuner = kt.RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='lstm_tuner',
        project_name='stock_prediction',
    )

    tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    return model, tuner

# Étape 3 : Entraîner ou Charger un modèle existant
def train_or_load_model(X_train, y_train, model_path, window_size):
    if os.path.exists(model_path):
        print("Chargement du modèle existant...")
        model = load_model(model_path)
    else:
        print("Optimisation et création du modèle...")
        model, _ = optimize_model(X_train, y_train)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
        model.save(model_path)
    return model

# Étape 4 : Prédictions multi-steps
def predict_multi_steps(model, data, steps, scaler, window_size):
    predictions = []
    last_sequence = data[-window_size:]
    for _ in range(steps):
        input_sequence = last_sequence.reshape(1, window_size, 1)
        pred = model.predict(input_sequence)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Étape 5 : Évaluation et Visualisation
def evaluate_and_visualize(model, X_test, y_test, predictions, scaler, data):
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_original = scaler.inverse_transform(predictions)

    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mae = mean_absolute_error(y_test_original, predictions_original)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    # Visualisation
    plt.figure(figsize=(14, 6))
    plt.plot(data['Close'], label='Vrai Prix', color='blue', alpha=0.5)
    plt.plot(range(len(data) - len(predictions), len(data)), predictions_original, label='Prédictions', color='red')
    plt.title('Prédictions des prix des actions')
    plt.xlabel('Temps')
    plt.ylabel('Prix de clôture')
    plt.legend()
    plt.show()

# Étape 6 : Pipeline complet
file_path = 'historical_stock_data.csv'
window_size = 60
model_path = 'stock_price_prediction_model.h5'

data, train_data, test_data, scaler = load_and_preprocess_data(file_path, window_size=window_size)

X_train, y_train = create_sequences(train_data, window_size)
X_test, y_test = create_sequences(test_data, window_size)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = train_or_load_model(X_train, y_train, model_path, window_size)

# Prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Évaluation et visualisation
evaluate_and_visualize(model, X_test, y_test, predictions, scaler, data)

# Prédiction pour plusieurs étapes dans le futur
future_steps = 30
future_predictions = predict_multi_steps(model, test_data, future_steps, scaler, window_size)
print(f"Prédictions pour les {future_steps} prochains jours :")
print(future_predictions)
