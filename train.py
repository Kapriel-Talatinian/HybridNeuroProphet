import numpy as np
from model import CNN1D
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)

def prepare_data(data, sequence_length=30):
   """Prépare les données historiques pour l'entraînement"""
   
   features = {
       'prices': data['close'].values,
       'volumes': data['volume'].values,
       'high_prices': data['high'].values,
       'low_prices': data['low'].values
   }
   
   # Calcul des indicateurs techniques
   prices = features['prices']
   
   # Supertrend
   atr_period = 10
   multiplier = 3
   hl2 = (features['high_prices'] + features['low_prices']) / 2
   tr = np.maximum(features['high_prices'] - features['low_prices'],
                  np.abs(features['high_prices'] - np.roll(prices, 1)))
   atr = pd.Series(tr).rolling(atr_period).mean().values
   
   basic_upperband = hl2 + (multiplier * atr)
   basic_lowerband = hl2 - (multiplier * atr)
   
   features['supertrend_up'] = basic_upperband
   features['supertrend_down'] = basic_lowerband
   
   # EMA
   ema_period = 20
   features['ema'] = pd.Series(prices).ewm(span=ema_period).mean().values
   
   # RSI
   delta = np.diff(prices, prepend=prices[0])
   gain = np.where(delta > 0, delta, 0)
   loss = np.where(delta < 0, -delta, 0)
   avg_gain = pd.Series(gain).rolling(14).mean().values
   avg_loss = pd.Series(loss).rolling(14).mean().values
   rs = avg_gain / (avg_loss + 1e-8)
   features['rsi'] = 100 - (100 / (1 + rs))

   # Création des séquences
   sequences = []
   labels = []
   
   for i in range(len(prices) - sequence_length):
       sequence = {}
       for key, values in features.items():
           sequence[key] = values[i:i+sequence_length]
           
       # Label: 1 si prix monte, 0 sinon
       label = 1 if prices[i+sequence_length] > prices[i+sequence_length-1] else 0
       
       sequences.append(sequence)
       labels.append(label)
       
   return sequences, np.array(labels)

def train_model(data_path, epochs=50, batch_size=32):
   logging.info("Chargement des données...")
   data = pd.read_csv(data_path)
   
   sequences, labels = prepare_data(data)
   logging.info(f"Séquences créées: {len(sequences)}")
   
   # Split train/validation
   split = int(len(sequences) * 0.8)
   train_seq, val_seq = sequences[:split], sequences[split:]
   train_labels, val_labels = labels[:split], labels[split:]
   
   # Initialisation modèle
   model = CNN1D()
   
   logging.info("Début entrainement...")
   best_val_accuracy = 0
   for epoch in range(epochs):
       # Shuffle training data
       idx = np.random.permutation(len(train_seq))
       train_seq = [train_seq[i] for i in idx]
       train_labels = train_labels[idx]
       
       # Training par batch
       for i in range(0, len(train_seq), batch_size):
           batch_seq = train_seq[i:i+batch_size]
           batch_labels = train_labels[i:i+batch_size]
           
           # Normalisation et stacking des features
           batch_features = []
           for seq in batch_seq:
               features = []
               for key in seq:
                   feature = seq[key]
                   normalized = (feature - np.mean(feature)) / (np.std(feature) + 1e-8)
                   features.append(normalized)
               batch_features.append(np.stack(features))
               
           batch_X = np.stack(batch_features)
           model.train_increment(batch_X, batch_labels)
           
           if i % 100 == 0:
               accuracy = model.training_history[-1]['accuracy']
               logging.info(f"Epoch {epoch+1}, Batch {i//batch_size}, Accuracy: {accuracy:.4f}")
       
       # Validation
       val_predictions = []
       for seq in val_seq:
           features = []
           for key in seq:
               feature = seq[key]
               normalized = (feature - np.mean(feature)) / (np.std(feature) + 1e-8)
               features.append(normalized)
           X = np.stack(features).reshape(1, -1, len(feature))
           pred = model.predict(X)[0]
           val_predictions.append(pred)
           
       val_accuracy = np.mean(val_predictions == val_labels)
       logging.info(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.4f}")
       
       if val_accuracy > best_val_accuracy:
           best_val_accuracy = val_accuracy
           model.save('best_model.joblib')
           
       if val_accuracy > 0.75:
           logging.info("Précision cible atteinte")
           break
   
   logging.info(f"Entrainement terminé. Meilleure précision validation: {best_val_accuracy:.4f}")
   return model

if __name__ == '__main__':
   train_model('historical_data.csv')