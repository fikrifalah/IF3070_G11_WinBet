import pickle
import numpy as np
import matplotlib.pyplot as plt


class linear_regression():
    def __init__(self, penalty=None, lmbda=0.01, method='gradient_descent', batch_size=None):
        """
        Parameters:
        - penalty: None, 'l2' (Ridge), atau 'l1' (Lasso)
        - lmbda: kekuatan regularisasi (float)
        - method: 'gradient_descent' atau 'normal_equation'
        - batch_size: None (Batch GD) atau int (Mini-Batch GD)
        """
        self.penalty = penalty
        self.lmbda = lmbda
        self.method = method
        self.batch_size = batch_size

    def gradient_descend(self, features: np.array, target: np.array, weights: np.array, max_iter=10000, learning_rate=0.001, tol=1e-5):
        n = features.shape[0]
        self.loss_history = []
        rng = np.random.default_rng(42)

        for step in range(max_iter):
            
            if self.batch_size is not None and self.batch_size < n:
                indices = rng.choice(n, size=self.batch_size, replace=False)
                X_batch = features[indices]
                y_batch = target[indices]
                n_batch = self.batch_size
            else:
                X_batch = features
                y_batch = target
                n_batch = n

            predicted_values = X_batch @ weights
            error = predicted_values - y_batch

            
            mse_loss = np.mean(error ** 2)
            self.loss_history.append(mse_loss)

            
            w_penalty = weights.copy()
            w_penalty[0] = 0.0

            gradient = (2 / n_batch) * (X_batch.T @ error)

            # Regularisasi L2 (Ridge)
            if self.penalty == 'l2':
                gradient += 2 * self.lmbda * w_penalty
            # Regularisasi L1 (Lasso)
            elif self.penalty == 'l1':
                gradient += self.lmbda * np.sign(w_penalty)

            
            if np.linalg.norm(gradient) < tol:
                break

            # Update weights setelah penalti ditambahkan ke gradient
            step_size = gradient * learning_rate
            weights = weights - step_size

        return weights

    def normal_equation(self, features: np.array, target: np.array):
        
        d = features.shape[1]
        if self.penalty == 'l2':
            # (X^T X + n * lambda * I)^(-1) X^T y, dengan I[0,0] = 0 agar bias tidak diregularisasi
            I = np.eye(d)
            I[0, 0] = 0.0
            n = features.shape[0]
            return np.linalg.pinv(features.T @ features + n * self.lmbda * I) @ (features.T @ target)
        else:
            return np.linalg.pinv(features.T @ features) @ (features.T @ target)

    def fit(self, features: np.array, target: np.array, max_iter=10000, learning_rate=0.001, tol=1e-5):
        self.features = np.column_stack([np.ones(features.shape[0]), features])
        self.target = target

        if self.method == 'normal_equation':
            self.weights = self.normal_equation(self.features, self.target)
            pred = self.features @ self.weights
            self.loss_history = [np.mean((pred - self.target) ** 2)]
        else:
            rng = np.random.default_rng(42)
            shape = (self.features.shape[1],)
            self.weights = rng.random(shape)
            self.weights = self.gradient_descend(
                self.features, self.target, self.weights,
                max_iter=max_iter, learning_rate=learning_rate, tol=tol
            )

    def predict(self, features):
        features_with_bias = np.column_stack([np.ones(features.shape[0]), features])
        return features_with_bias @ self.weights

    def plot_loss(self, title='Loss Curve per Iterasi'):
        """Plot loss curve over iterations."""
        if not hasattr(self, 'loss_history') or len(self.loss_history) == 0:
            print("Model belum di-fit!")
            return
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label='Loss (MSE)', color='blue')
        plt.title(title)
        plt.xlabel('Iterasi / Step')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.legend()
        plt.show()

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model berhasil disimpan ke {filepath}")

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model berhasil dimuat dari {filepath}")
        return model
