#ReadMe
''''**Co-Design Optimization Logic - Gradient Descent Version**

This is a sample logic code to demonstrate the feasibility of co-optimization. Users can personalize the design based on this logic (linking with various functions) and deploy it to specific IoT systems for optimization.

Users can replace `*` with simple values to run the code and understand the logic.
'''



import numpy as np


# Function to calculate total system latency
def compute_total_latency(lambda_val, I_0, N, mu_1, mu_2, beta_1, beta_2, beta_s, N_e):
    # Calculate computational latency for submodels 1 and 2
    T_1 = (lambda_val * N) / (mu_1 * I_0 * beta_1)
    T_2 = ((1 - lambda_val) * N) / (mu_2 * I_0 * beta_2)

    # Calculate data transmission latency
    T_3 = N_e / beta_s

    # Return total system latency
    return T_1 + T_2 + T_3


# Function to calculate total accuracy loss based on fusion strategy
def compute_accuracy_loss(theta_1, theta_2, delta_A_1, delta_A_2, lambda_val):
    A_1 = np.sum(theta_1 * delta_A_1)
    A_2 = np.sum(theta_2 * delta_A_2)
    total_accuracy_loss = lambda_val * A_1 + (1 - lambda_val) * A_2
    return total_accuracy_loss


# Joint optimization function using gradient descent for lambda_val, theta1, and theta2
def joint_optimization(I_0, N_values, mu_1, mu_2, beta_1, beta_2, beta_s, N_e, delta_A_1, delta_A_2,
                       lagrangian_multiplier, learning_rate=0.01, iterations=100):
    # Initial guesses
    lambda_val = 0.5  # Initial lambda_val
    theta1 = np.array([1, 0, 0, 0], dtype=float)  # Continuous theta1 initialized close to discrete options
    theta2 = np.array([1, 0, 0, 0], dtype=float)  # Continuous theta2 initialized close to discrete options

    best_latency = float('inf')
    best_lambda = None
    best_theta = None

    # Gradient descent loop
    for _ in range(iterations):
        # Calculate total system latency and accuracy loss
        total_latency = compute_total_latency(lambda_val, I_0, N_values, mu_1, mu_2, beta_1, beta_2, beta_s, N_e)
        total_accuracy_loss = compute_accuracy_loss(theta1, theta2, delta_A_1, delta_A_2, lambda_val)

        # Calculate joint cost
        joint_cost = total_latency + lagrangian_multiplier * total_accuracy_loss

        # Update best solution if current cost is lower
        if joint_cost < best_latency:
            best_latency = joint_cost
            best_lambda = lambda_val
            best_theta = (theta1.copy(), theta2.copy())

        # Compute gradients with respect to lambda_val, theta1, and theta2
        grad_latency_lambda = (N_values / (mu_1 * I_0 * beta_1)) - (N_values / (mu_2 * I_0 * beta_2))
        grad_accuracy_loss_lambda = np.sum(delta_A_1 * theta1) - np.sum(delta_A_2 * theta2)
        grad_lambda_val = grad_latency_lambda + lagrangian_multiplier * grad_accuracy_loss_lambda

        grad_accuracy_loss_theta1 = lambda_val * delta_A_1
        grad_accuracy_loss_theta2 = (1 - lambda_val) * delta_A_2

        # Update lambda_val, theta1, and theta2 using gradient descent
        lambda_val -= learning_rate * grad_lambda_val
        theta1 -= learning_rate * grad_accuracy_loss_theta1
        theta2 -= learning_rate * grad_accuracy_loss_theta2

        # Clamp lambda_val to [0, 1] to maintain validity, and theta values to [0, 1]
        lambda_val = max(0, min(1, lambda_val))
        theta1 = np.clip(theta1, 0, 1)
        theta2 = np.clip(theta2, 0, 1)

    return best_lambda, best_theta, best_latency


# Define input parameters
I_0 = *
N_values = *
mu_1, mu_2 = *
beta_1, beta_2, beta_s = *
N_e = *
delta_A_1 = np.array(*)
delta_A_2 = np.array(*)
lagrangian_multiplier = *

# Run the joint optimization function
best_lambda, best_theta, best_latency = joint_optimization(I_0, N_values, mu_1, mu_2, beta_1, beta_2, beta_s, N_e,
                                                           delta_A_1, delta_A_2, lagrangian_multiplier)

# Output the optimal results
print(f"Optimal split point: {best_lambda}")
print(f"Optimal strategy combination: {best_theta}")
print(f"Minimum system latency: {best_latency}")
