import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle


# Define the number of epochs, initial learning rate, and warm-up epochs
num_epochs = 20
initial_lr = 0.1
warmup_epochs = 5

# Create a dummy optimizer and initialize the learning rate
optimizer = optim.SGD([torch.randn(1, requires_grad=True)], lr=initial_lr)
lr = initial_lr

# Lists to store the learning rate and epoch values
learning_rates = []
epochs = []

# Simulate the training epochs
for epoch in range(num_epochs):
    # Perform warm-up for the first few epochs
    if epoch < warmup_epochs:
        lr = (initial_lr / warmup_epochs) * (epoch + 1)
    else:
        # Calculate the progress within the cosine annealing phase
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        lr = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415)))

    # Log the current learning rate
    print(f"Epoch {epoch + 1}: Learning Rate = {lr}")

    # Store the learning rate and epoch values
    learning_rates.append(lr)
    epochs.append(epoch)

# Plot the learning rate values
plt.plot(epochs, learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()