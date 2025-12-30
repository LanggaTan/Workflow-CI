import mlflow
import mlflow.keras
from mlflow.models import infer_signature
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import optuna
import dagshub

script_dir = os.path.dirname(os.path.abspath(__file__))
X_dir = os.path.join(script_dir, 'Dataset_preprocess/X_timestep_10.npy')
y_dir = os.path.join(script_dir, "Dataset_preprocess/y_timestep_10.npy")
X = np.load(X_dir)
y = np.load(y_dir)
sample_count, timesteps, feature_count = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, input_shape=(timesteps, feature_count)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

mlflow.tensorflow.autolog()

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("lstm_units", 64)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(timesteps, feature_count)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    for i, loss in enumerate(history.history['loss']):
        mlflow.log_metric("train_loss", loss, step=i)
    for i, val_loss in enumerate(history.history['val_loss']):
        mlflow.log_metric("val_loss", val_loss, step=i)

    #Artifact
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss.png")
    plt.close()

    mlflow.log_artifact("loss.png")
    mlflow.keras.log_model(model, name="lstm_model")
    mlflow.log_text(str(history.history), "history.txt")
