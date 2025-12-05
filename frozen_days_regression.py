import sys
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

#########################################
NUM_ITERS = 200
#########################################

if __name__ == "__main__":
    #Read in inputs
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    x = df["year"].values
    y = df["days"].values


    #Visualize Data
    plt.figure()                  
    plt.plot(x, y, marker='o')
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.xticks(x)
    plt.savefig("data_plot.jpg")


    #Data Normalization
    m = float(np.min(x))
    M = float(np.max(x))
    x_tilde = (x - m) / (M - m)

    ones = np.ones_like(x_tilde)
    X_normalized = np.column_stack([x_tilde, ones])

    print("X_normalized:")
    print(X_normalized)

    
    #Linear Regression w/ Closed Form Solution

    Y = y.reshape(-1, 1)
    X_tilde = X_normalized
    weights = np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ Y

    print("Weights flattened:")
    print(weights.flatten())
    
    #Linear Regression w/ Gradient Descent

    LEARNING_RATE = 0.33
    
    gd_X = torch.tensor(X_normalized, dtype=torch.float64)
    gd_Y = torch.tensor(Y, dtype=torch.float64)
    gd_weights = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    n = len(gd_Y)
    losses = np.zeros(NUM_ITERS)
    
    print("Val:")
    for iter in range(NUM_ITERS):
        if iter % 20 == 0:
            print(gd_weights.detach().numpy())
        loss = torch.mean(((gd_X @ gd_weights) - gd_Y.squeeze()) ** 2)
        
        losses[iter] = loss.item()
        
        #Performs a backward pass through the computation graph
        #After this line, the gradient of the loss with respect to the weights is in gd_weights.grad
        loss.backward()

        #Performs one step of gradient descent
        with torch.no_grad():
            gd_weights -= LEARNING_RATE * gd_weights.grad

        #Resets the computation graph
        gd_weights.grad.zero_()

    plt.figure()
    plt.plot(range(NUM_ITERS), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg") 

    print("Learning rate:", LEARNING_RATE)
    print("How to determine lr: I started with 0.01 and increased it to 0.33 until it converged close to the closed-form result.")
    
    #Prediction
    w = weights[0][0] 
    b = weights[1][0] 
    xTest = 2024 

    xTest_tilde = (xTest - m) / (M - m)
    X_test = np.array([[xTest_tilde, 1.0]])
    y_hat = float((X_test @ weights)[0, 0])

    print("y_hat: " + str(y_hat))


    #Model Interpretation
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="
    
    print("Symbol: " + symbol)
    print("Explaination: w>0 means ice days increase over years. w<0 means ice days decrease. w=0 means there is no clear trend.")


    # Model Limitations
    x_star = m - (b * (M - m)) / w
    print("x_star: " + str(x_star))
    print("Accurate? This prediction is not reliable because the model is linear while the real pattern is more complex. Other factors can make this prediction inaccurate.")