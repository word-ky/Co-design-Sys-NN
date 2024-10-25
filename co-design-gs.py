#ReadMe
''''**Co-Design Optimization Logic - Grid Search Version**

This is a sample logic code to demonstrate the feasibility of co-optimization. Users can personalize the design based on this logic (linking with various functions) and deploy it to specific IoT systems for optimization.

Users can replace `*` with simple values to run the code and understand the logic.
'''


import numpy as np


# Function to calculate total system latency
def compute_total_latency(lambda_val, k, p, I_0, N, mu_1, mu_2, beta_1, beta_2, beta_s, N_e):
    # Calculate lambda_C and lambda_M
    lambda_C = k * lambda_val
    lambda_M = p * lambda_val

    # Calculate I_1 and I_2
    I_1 = (lambda_C / lambda_M) * I_0
    I_2 = ((1 - lambda_C) / (1 - lambda_M)) * I_0

    # Calculate computational latency for submodels 1 and 2
    T_1 = (lambda_val * N) / (mu_1 * I_1 * beta_1)
    T_2 = ((1 - lambda_val) * N) / (mu_2 * I_2 * beta_2)

    # Calculate data transmission latency
    T_3 = N_e / beta_s

    # Return total system latency
    return T_1 + T_2 + T_3


# Function to calculate total accuracy loss based on fusion strategy
def compute_accuracy_loss(theta_1, theta_2, delta_A_1, delta_A_2, lambda_val):
    A_1 = np.sum([sign * delta for sign, delta in zip(theta_1, delta_A_1)])
    A_2 = np.sum([sign * delta for sign, delta in zip(theta_2, delta_A_2)])
    total_accuracy_loss = lambda_val * A_1 + (1 - lambda_val) * A_2
    return total_accuracy_loss


# Joint optimization function
def joint_optimization(lambda_values, theta_values, I_0, N_values, mu_1, mu_2, beta_1, beta_2, beta_s, N_e, pi_1, pi_2,
                       delta_A_1, delta_A_2, lagrangian_multiplier):
    # Calculate I_m1 and I_m2
    I_m1 = pi_1 / beta_1
    I_m2 = pi_2 / beta_2

    best_latency = float('inf')
    best_lambda = None
    best_theta = None

    # Iterate over all possible lambda values
    for lambda_val in lambda_values:
        # Iterate over all possible theta combinations
        for theta1 in theta_values[0]:
            for theta2 in theta_values[1]:
                # Automatically search for k and p values
                #for k in np.linspace(0.1, 2.0, 20):
                    #for p in np.linspace(0.1, 2.0, 20):
                        # Calculate lambda_C and lambda_M
                        lambda_C = k * lambda_val
                        lambda_M = p * lambda_val

                        # Calculate I_1 and I_2
                        I_1 = (lambda_C / lambda_M) * I_0
                        I_2 = ((1 - lambda_C) / (1 - lambda_M)) * I_0

                        # Check if I_1 and I_2 meet constraint conditions
                        #if I_1 >= I_m1 and I_2 >= I_m2:
                        print(f"Constraints met: lambda_val={lambda_val}, I_1={I_1}, I_2={I_2}, k={k}, p={p}")

                        # Calculate total system latency
                        total_latency = compute_total_latency(lambda_val, k, p, I_0, N_values, mu_1, mu_2, beta_1,
                                                                  beta_2, beta_s, N_e)

                        # Calculate accuracy loss
                        total_accuracy_loss = compute_accuracy_loss(theta1, theta2, delta_A_1, delta_A_2,
                                                                        lambda_val)

                        # Calculate joint optimization objective function value
                        joint_cost = total_latency + lagrangian_multiplier * total_accuracy_loss

                        # Update the best solution
                        if joint_cost < best_latency:
                            best_latency = joint_cost
                            best_lambda = lambda_val
                            best_theta = (theta1, theta2)

    return best_lambda, best_theta, best_latency


# Define input parameters
lambda_values =*
theta_values =*
I_0 =*
N_values =*
mu_1, mu_2 =*
beta_1, beta_2, beta_s =*
N_e =*
pi_1, pi_2 =*
delta_A_1 =*
delta_A_2 =*
lagrangian_multiplier =*

# Run the joint optimization function
best_lambda, best_theta, best_latency = joint_optimization(lambda_values, theta_values, I_0, N_values, mu_1, mu_2,
                                                           beta_1, beta_2, beta_s, N_e, pi_1, pi_2, delta_A_1,
                                                           delta_A_2, lagrangian_multiplier)

# Output the optimal results
print(f"Optimal split point: {best_lambda}")
print(f"Optimal strategy combination: {best_theta}")
print(f"Minimum system latency: {best_latency}")
