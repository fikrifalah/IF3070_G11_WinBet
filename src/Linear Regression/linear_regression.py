
import pickle
import numpy as np
import matplotlib.pyplot as plt

class linear_regression():
    def __init__(self, penalty=None, lmbda=0.01):
        self.penalty = penalty   
        self.lmbda = lmbda 
        
        # self.target_matrix = target
        # self.feature_matrix = np.column_stack([np.ones(features.shape[0]), features])
        # self.weights = np.random()
        
    # def forward(self, percept, slope, feature):
    #     return percept + slope * feature
    
    def gradient_descend(self,  features: np.array, target: np.array, weights: np.array, max_iter = 10000,learning_rate = 0.001, tol = 0.00001):
        n = features.shape[0]
        self.loss_history = []
       
        for step in range(max_iter):
            predicted_values = features @ weights
            error = predicted_values - target

            # Compute MSE loss
            mse_loss = np.mean(error ** 2)
            self.loss_history.append(mse_loss)

            w_penalty = weights.copy()
            w_penalty[0] = 0

            gradient = (2/n) * (features.T @ error) 
            step_size = gradient * learning_rate

            # If L2 (Ridge):
            if self.penalty == 'l2':
                gradient += 2 * self.lmbda * w_penalty
            # If L1 (Lasso):
            elif self.penalty == 'l1':
                gradient += self.lmbda * np.sign(w_penalty)

            if(np.linalg.norm(gradient) < tol):
                break
            weights = weights - step_size
        return weights
    
    def fit(self, features: np.array, target:np.array):
        self.features = np.column_stack([np.ones(features.shape[0]), features])
        self.target = target
        rng = np.random.default_rng()
        shape = (self.features.shape[1],)

        self.weights = rng.random(shape)
    
        self.weights = self.gradient_descend(self.features, self.target, self.weights)


       
    def predict(self, features):
        features_with_bias = np.column_stack([np.ones(features.shape[0]), features])
        return features_with_bias @ self.weights

    def plot_loss(self):
        """Plot loss curve over iterations."""
        if not hasattr(self, 'loss_history') or len(self.loss_history) == 0:
            print("Model belum di-fit!")
            return
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label='Loss (MSE)', color='blue')
        plt.title('Loss Curve per Iterasi')
        plt.xlabel('Iterasi / Step')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.legend()
        plt.show()

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model berhasil disimpan ke {filepath}")

    
    def load(cls, filepath: str):
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model berhasil dimuat dari {filepath}")
        return model





