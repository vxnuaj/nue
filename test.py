import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss

def log_loss(y, pred, eps=1e-8):
    loss = - (np.sum(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))) / y.size
    return loss

# Example usage
y = np.array([0, 1, 1, 0])
pred = np.array([0.1, 0.9, 0.8, 0.4])

# Your log loss implementation
my_loss = log_loss(y, pred)
print(f"My Log Loss: {my_loss}")

# Sklearn log loss implementation
sklearn_loss = sklearn_log_loss(y, pred)
print(f"Sklearn Log Loss: {sklearn_loss}")