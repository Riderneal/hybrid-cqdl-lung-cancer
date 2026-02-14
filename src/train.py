import torch
import torch.nn as nn
import torch.optim as optim
from hybrid_model import HybridModel

# Dummy dataset (for structure testing)
X = torch.randn(32, 1, 128, 128)
y = torch.randint(0, 2, (32, 1)).float()

model = HybridModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    optimizer.zero_grad()
    
    outputs = model(X)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
