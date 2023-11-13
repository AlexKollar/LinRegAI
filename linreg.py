import torch
import os 
import pandas as pd 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from pathlib import Path

#Create Knonwn parameters:
weight = 0.7
bias = 0.3

#create the basic data for the tensors. 
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

trainSplit = int(0.8 * len(X))
X_train, y_train = X[:trainSplit], y[:trainSplit]
X_test, y_test = X[trainSplit:], y[trainSplit:]
#print(len(X_train), len(y_train), len(X_test), len(y_test))

def plotPrediction(trainData = X_train,
                   trainLables = y_train,
                   testData = X_test,
                   testLables = y_test,
                   predictions = None):
    """
    Plots training data, test data and compares predictions 
    """
    plt.figure(figsize=(10 , 7))
    plt.scatter(trainData, trainLables, c = "b", s = 4, label = "Training Data")
    plt.scatter(testData, testLables, c = "g", s = 4, label = "Test Data")

    if predictions is not None:
        plt.scatter(testData, predictions, c = "r", s =4, label = "Predictions")

    plt.legend(prop={"size": 14});
    plt.show()

"""
Lets talk linear regression Formula: Y = a + bX
This is best shown off later on the LinearRegressionModel class forward def. 
"""

#Create linear regression model:
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias 

"""
Lets do some prediction with torch.inference_mode()
to check our models predictive power, lets see how well it predicts 'y_test' based on 'x_test'
when we pass data through the forward() 
"""

torch.manual_seed(42)
model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(X_test)
print("Plotting initial predictions: ")
plotPrediction(predictions=y_preds)

"""
3. Training the model to do better than the above. 
The whole idea of training an AI Model is to move from *unknown parameters* to some *known* parameters
In other words from a poor represenation to an accurate representation.

One way to measure how poor predictions are is to use a loss function. 
Note: Loss function may also be called cost function.

Things we need to train: 
- Loss Function: A function to measure how bad your predictions are. The lower the value
the better.
- Optimizer: Adjusts the weight and bias to improve the loss function. 

And Specifically for pytorch we need: 
- A training loop
- A test loop 
"""

#Setup Loss function:
loss_fn = nn.L1Loss()

#Setup Optimizer:
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# Training Loop
epochs = int(input("\nNow lets train the model to get better results!\n\nHow many Epochs do you want to run: ")) 
print(f"\nTraining Model with {epochs} epochs")
# 0. Loop through the data:
for epochs in range(epochs):
    #set model to training mode
    model_0.train()
    # 1. Forward pass
    y_preds = model_0(X_train)
    # 2. Calculate the loss
    loss = loss_fn(y_preds, y_train)
    #print(f"Loss: {loss}")
    # 3. optimizer zero grad
    optimizer.zero_grad()
    # 4. Backpropagation
    loss.backward()
    # 5. Step optimizer
    optimizer.step()
    
    #testing Loop
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model_0(X_test)
    
        # 2. Calculate Loss:
        test_loss = loss_fn(test_pred, y_test)
    
    if epochs % 10 == 0:
        #os.system("clear")
        print(f"Epoch: {epochs} | Loss: {loss} | Test Loss: {test_loss} |  ")     
        print(f"Current State:\n{model_0.state_dict()}\n")


#Plot the new predictions
input("\nPress enter to plot the trained predictions: ")
with torch.inference_mode():
    new_preds = model_0(X_test)

plotPrediction(predictions=new_preds)

saveModelChoice = input("Do you want to save the model\n(yes or no): ").lower()
if saveModelChoice == "yes":
    #Create Model Directory: 
    MODEL_PATH = Path("Models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    #Create Model Save Path
    MODEL_NAME = "01_Linear.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
    print("Model has been saved.")
elif saveModelChoice == "no":
    print("Okay, thanks for checking this out.")
else:
    print("Option invalid moving on...")
