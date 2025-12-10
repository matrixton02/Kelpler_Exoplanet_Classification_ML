import numpy as np

def sigmoid(z):                                                     # this is our sigmoid fucntiion which squeezes any value between 0 or 1 
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

def compute_cost(theta, X, y, lambda_reg=0):                          # this fucntion computes the cost
    m = y.size
    h = sigmoid(X @ theta)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    if lambda_reg > 0:
        cost += (lambda_reg / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost

def gradient(theta, X, y, lambda_reg=0):                            # this provides the gradient i have added L2 regularization to it 
    m = y.size
    grad = (X.T @ (sigmoid(X @ theta) - y)) / m
    
    # Add regularization (excluding bias term)
    if lambda_reg > 0:
        reg_term = np.zeros_like(theta)
        reg_term[1:] = (lambda_reg / m) * theta[1:]
        grad += reg_term
    
    return grad

def gradient_descent(X, y, alpha=0.1, num_iter=100, lambda_reg=0, tol=1e-7, verbose=True):          # Gradient descent fucntion which perform the gradient decent algorithm for training out model and find the right weights
    """
    Gradient descent with optional regularization and convergence monitoring
    
    Parameters:
    -----------
    X : array, shape (m, n) - features
    y : array, shape (m,) - labels
    alpha : float - learning rate
    num_iter : int - maximum iterations
    lambda_reg : float - L2 regularization parameter
    tol : float - convergence tolerance
    verbose : bool - print training progress
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])
    
    cost_history = []
    for i in range(num_iter):
        grad = gradient(theta, X_b, y, lambda_reg)
        theta -= alpha * grad
        
        if i % 100 == 0 or i == num_iter - 1:
            cost = compute_cost(theta, X_b, y, lambda_reg)
            cost_history.append(cost)
            if verbose and i % (num_iter/100) == 0:
                print(f"Iteration {i:5d}: Cost = {cost:.6f}, |grad| = {np.linalg.norm(grad):.6f}")
        
        if np.linalg.norm(grad) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
    
    return theta, cost_history

def predict_prob(X, theta):                                                     # predicts the output of any input data 
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

def predict(X, theta, threshold=0.5):                                           # this fucntion is called for predicting the output of any data
    return (predict_prob(X, theta) >= threshold).astype(int)
