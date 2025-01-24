from flask import Flask, request, jsonify
import numpy as np
from model import CNN1D
import joblib

app = Flask(__name__)
model = CNN1D(input_channels=7, sequence_length=30)  # 7 indicateurs

@app.route('/train', methods=['POST'])
def handle_training():
    data = request.json
    
    # Convertir les données
    X = np.array(data['Features'])
    X = X.reshape(1, 7, 30)  # (batch_size, features, sequence_length)
    
    # Générer un label (ex: prédiction de tendance)
    y = np.array([1 if X[0, 3, -1] > X[0, 0, -1] else 0])  # Close > Open
    
    # Entraînement incrémental
    model.train_increment(X, y, epochs=1)
    
    # Sauvegarder le modèle
    joblib.dump(model, 'trained_model.joblib')
    
    return jsonify({"status": "success", "accuracy": model.best_accuracy})