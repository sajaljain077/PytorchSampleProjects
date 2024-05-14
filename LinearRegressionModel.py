


import torch
from torch import nn
import matplotlib.pyplot as plt
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

weight = 0.7
bias = 0.3

start =0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias



# Train and test data creation
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split].to(device), y[:train_split].to(device)
X_test, y_test = X[train_split:].to(device), y[train_split:].to(device)


# Code for data visualization
def plot_prediction(train_data, train_labels, test_data, test_labels,
                    prediction = None):
    plt.figure(figsize=(14,7))
    plt.scatter(train_data, train_labels, c='b', label = "Training data")
    plt.scatter(test_data, test_labels, c = 'g', label = "Test data")
    if prediction is not None:
        plt.scatter(test_data, prediction, c = 'r', label = "predicted Test data")
    plt.legend()
    plt.show()

# plot_prediction(X_train, y_train, X_test, y_test)


# Linear Regression Model
class Linear_Regression_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(1),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1),
                                    requires_grad = True)
        
    def forward(self, x:torch.tensor) -> torch.tensor:
        return self.weight *  x + self.bias




# This is to check the parameters for the model
"""
    model_0 = Linear_Regression_Model()
    print(list(model_0.parameters())) 
    print(model_0.state_dict())
"""


"""
#To predict the current model output we can do this
model_0 = Linear_Regression_Model()
with torch.inference_mode():
    prediction = model_0(X_test)
# plot_prediction(X_train, y_train, X_test, y_test, prediction)

"""






model_0 = Linear_Regression_Model()
model_0.to(device)
# Loss function and optimizer information
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
epochs = 200


train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()  # this is to enable the train mode
    y_pred = model_0(X_train) # this is to forward pass
    loss = loss_fn(y_pred, y_train) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.cpu().detach().numpy())
            test_loss_values.append(test_loss.cpu().detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

print(list(model_0.parameters()))

# 1. Set the model in evaluation mode
model_0.eval()
# 2. Setup the inference mode context manager
with torch.inference_mode():
  y_preds = model_0(X_test)


plot_prediction(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), y_preds.cpu())
