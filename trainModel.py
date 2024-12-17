# Custom Dataset to Iterate Over Real-Space Pixels
class RealSpacePixelDataset(Dataset):
    def __init__(self, pixel_param_data, fourD_data):
        #self.real_space_data = real_space_data  # Shape: (59, 59)
        self.pixel_param_data = pixel_param_data  # Shape: (59, 59)
        self.fourD_data = fourD_data  # Shape: (59, 59, 64, 64)

    def __len__(self):
        return self.fourD_data.shape[0] * self.fourD_data.shape[1]

    def __getitem__(self, idx):
        row = idx // self.fourD_data.shape[1]
        col = idx % self.fourD_data.shape[1]
        
        pixel_param = self.pixel_param_data[row, col]  # Scalar value
        diffraction = self.fourD_data[row, col,:,:]  # Shape: (64, 64)
        
        # Combine pixel parameters and diffraction data into a 2-channel input
        input_tensor = torch.stack([torch.full_like(diffraction, pixel_param), diffraction], dim=0)
        return input_tensor, (row, col)

# Instantiate the model
model = IntoInelastic2D()

# Example dataset inputs
#real_space_data = torch.randn(59, 59)
pixel_param_data = torch.randn(59, 59)
fourD_data = torch.randn(59, 59, 64, 64)

# Prepare dataset and dataloader
dataset = RealSpacePixelDataset(pixel_param_data, fourD_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)


# Forward pass through each pixel
model.eval()
output_data = torch.zeros(59, 59, 64, 64)  # Output tensor placeholder
for input_tensor, (row, col) in dataloader:
    with torch.no_grad():
        output = model(input_tensor)  # Input shape: (1, 2, 64, 64)
        output_data[row, col] = output.squeeze()

print("Output shape:", output_data.shape)  # Should be (59, 59, 64, 64)
