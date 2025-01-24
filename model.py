import numpy as np
import joblib

class CNN1D:
    def __init__(self, input_channels=7, sequence_length=30):
        print(f"Initializing CNN with {input_channels} input channels and {sequence_length} sequence length")
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        
        # Hyperparamètres optimisés
        self.learning_rate = 0.01
        self.batch_size = 32
        self.momentum = 0.9
        
        # Architecture améliorée
        n_filters1 = 32
        n_filters2 = 64
        kernel_size = 3
        
        # Initialisation des poids avec normalisation
        self.conv1_filters = np.random.randn(n_filters1, input_channels, kernel_size) * np.sqrt(2.0/input_channels)
        self.conv2_filters = np.random.randn(n_filters2, n_filters1, kernel_size) * np.sqrt(2.0/n_filters1)
        
        conv1_output_length = sequence_length - kernel_size + 1
        conv2_output_length = conv1_output_length - kernel_size + 1
        
        self.fc1_weights = np.random.randn(n_filters2 * conv2_output_length, 128) * np.sqrt(2.0/(n_filters2 * conv2_output_length))
        self.fc2_weights = np.random.randn(128, 1) * np.sqrt(2.0/128)
        
        # Métriques de performance
        self.training_history = []
        self.total_updates = 0
        self.best_accuracy = 0

    def sanitize(self, x):
        """Remplace les valeurs non finies (NaN, Inf) par des valeurs par défaut."""
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    def conv1d(self, x, filters):
        """Applique une convolution 1D."""
        batch_size, in_channels, seq_len = x.shape
        n_filters, filter_channels, kernel_size = filters.shape
        output_length = seq_len - kernel_size + 1
        output = np.zeros((batch_size, n_filters, output_length))
        
        for b in range(batch_size):
            for f in range(n_filters):
                for i in range(output_length):
                    window = x[b, :, i:i+kernel_size]
                    output[b, f, i] = np.sum(window * filters[f])
        return self.sanitize(output)

    def relu(self, x):
        """Fonction d'activation ReLU."""
        return self.sanitize(np.maximum(0, x))

    def sigmoid(self, x):
        """Fonction d'activation sigmoïde."""
        return self.sanitize(1 / (1 + np.exp(-np.clip(x, -100, 100))))

    def forward(self, x):
        """Passe avant du réseau de neurones."""
        x = self.sanitize(x)
        
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
            
        # Garder les activations pour le backward pass
        self.layer_outputs = []
        
        # Couche de convolution 1
        conv1 = self.conv1d(x, self.conv1_filters)
        relu1 = self.relu(conv1)
        self.layer_outputs.append((conv1, relu1))
        
        # Couche de convolution 2
        conv2 = self.conv1d(relu1, self.conv2_filters)
        relu2 = self.relu(conv2)
        self.layer_outputs.append((conv2, relu2))
        
        # Aplatir les données pour la couche fully connected
        flattened = self.sanitize(relu2.reshape(x.shape[0], -1))
        self.layer_outputs.append(flattened)
        
        # Couche fully connected 1
        fc1 = np.dot(flattened, self.fc1_weights)
        relu3 = self.relu(fc1)
        self.layer_outputs.append((fc1, relu3))
        
        # Couche fully connected 2
        fc2 = np.dot(relu3, self.fc2_weights)
        output = self.sigmoid(fc2)
        
        return self.sanitize(output)

    def update_weights(self, gradient):
        """Mise à jour des poids avec momentum."""
        if not hasattr(self, 'weight_updates'):
            self.weight_updates = {
                'conv1': np.zeros_like(self.conv1_filters),
                'conv2': np.zeros_like(self.conv2_filters),
                'fc1': np.zeros_like(self.fc1_weights),
                'fc2': np.zeros_like(self.fc2_weights)
            }
        
        # Mise à jour avec momentum
        self.weight_updates['conv1'] = self.momentum * self.weight_updates['conv1'] - gradient * self.conv1_filters
        self.weight_updates['conv2'] = self.momentum * self.weight_updates['conv2'] - gradient * self.conv2_filters
        self.weight_updates['fc1'] = self.momentum * self.weight_updates['fc1'] - gradient * self.fc1_weights
        self.weight_updates['fc2'] = self.momentum * self.weight_updates['fc2'] - gradient * self.fc2_weights
        
        # Application des mises à jour
        self.conv1_filters += self.weight_updates['conv1']
        self.conv2_filters += self.weight_updates['conv2']
        self.fc1_weights += self.weight_updates['fc1']
        self.fc2_weights += self.weight_updates['fc2']
        
        self.total_updates += 1

    def train_increment(self, X, y, epochs=1):
        """Entraînement incrémental du modèle."""
        X = np.array(X)
        y = np.array(y)
        
        for epoch in range(epochs):
            predictions = self.forward(X)
            error = y - predictions.flatten()
            self.update_weights(error * self.learning_rate)
            
            # Calcul de la précision
            accuracy = np.mean((predictions.flatten() > 0.5) == y)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
            
            # Log des métriques
            self.training_history.append({
                'epoch': self.total_updates,
                'loss': np.mean(np.abs(error)),
                'accuracy': accuracy
            })

    def predict(self, x):
        """Fait une prédiction sur les données d'entrée."""
        output = self.forward(x)
        return output > 0.5, output

    def save(self, filename):
        """Sauvegarde le modèle dans un fichier."""
        weights = {
            'conv1': self.conv1_filters,
            'conv2': self.conv2_filters,
            'fc1': self.fc1_weights,
            'fc2': self.fc2_weights,
            'training_history': self.training_history,
            'best_accuracy': self.best_accuracy,
            'total_updates': self.total_updates
        }
        joblib.dump(weights, filename)

    def load(self, filename):
        """Charge le modèle à partir d'un fichier."""
        weights = joblib.load(filename)
        self.conv1_filters = weights['conv1']
        self.conv2_filters = weights['conv2']
        self.fc1_weights = weights['fc1']
        self.fc2_weights = weights['fc2']
        self.training_history = weights.get('training_history', [])
        self.best_accuracy = weights.get('best_accuracy', 0)
        self.total_updates = weights.get('total_updates', 0)