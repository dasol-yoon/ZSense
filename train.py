from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        self.input_images = input_images  # Tensor of (1) "elastic" image and (2) z values.
        self.target_images = target_images  # Tensor of (3) "inelastic" image
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        x = self.input_images[idx]
        y = self.target_images[idx]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

# Example usage
train_dataset = CustomImageDataset(input_images, target_images)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


import torch
import torch.optim as optim
import torch.nn as nn

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model = IntoInelastic().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Use L1Loss if needed
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20  # Set number of epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0.0

    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()

    # Print epoch summary
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

model.eval()  # Set model to evaluation mode
val_loss = 0.0

with torch.no_grad():
    for batch in val_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)
        val_loss += loss.item()

print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "unet_stage2.pth")

# model = IntoInelastic()
# model.load_state_dict(torch.load("IntoInelastic.pth"))
# model.to(device)
