import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import matplotlib.pyplot as plt
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.optim import Lamb

# self,
# optimizer: torch.optim.Optimizer,
# t_initial: int,
# lr_min: float = 0.,
# cycle_mul: float = 1.,
# cycle_decay: float = 1.,
# cycle_limit: int = 1,
# warmup_t=0,
# warmup_lr_init=0,
# warmup_prefix=False,
# t_in_epochs=True,
# noise_range_t=None,
# noise_pct=0.67,
# noise_std=1.0,
# noise_seed=42,
# k_decay=1.0,
# initialize=True,

# Define the number of epochs and the initial learning rate
num_epochs = 80
lr_0 = 1e-1
warmup_epochs = 5

# Create an optimizer and a learning rate scheduler
optimizer = timm.optim.Lamb([torch.randn(1, requires_grad=True)], lr=lr_0, weight_decay=0.02)
#optimizer = optim.SGD([torch.randn(1, requires_grad=True)], lr=lr_0)
#scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_0/100.0, last_epoch=- 1, verbose=False)
scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=num_epochs, lr_min=lr_0/10.0**3, warmup_t=5, warmup_lr_init=lr_0/10.0**2)
use_sched = True

# Lists to store the learning rate and epoch values
learning_rates = []
epochs = []

# Simulate the training epochs
for epoch in range(num_epochs):
    # Log the current learning rate
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}: Learning Rate = {lr}")

    # Store the learning rate and epoch values
    learning_rates.append(lr)
    epochs.append(epoch)

    # Update the learning rate
    optimizer.step()
    scheduler.step(epoch=epoch)

# Plot the learning rate values
plt.plot(epochs, learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()



