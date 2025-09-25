import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("=== SeaSeeAI Simple Demo ===")

# Simple model
class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x).view(batch_size, 5, 4)

# Create dummy data
X = torch.randn(100, 10, 4)
y = torch.randn(100, 5, 4)

model = DemoModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training simple model...")
losses = []
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

plt.plot(losses)
plt.title("SeaSeeAI Training Demo")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("demo_training.png")
print("Demo completed! Check demo_training.png")
