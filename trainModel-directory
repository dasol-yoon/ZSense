import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import os

class IntoInelastic2D(nn.Module):
    def __init__(self):
        super(IntoInelastic2D, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(2, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.dec3 = self.upconv_block(512 + 256, 256)
        self.dec2 = self.upconv_block(256 + 128, 128)
        self.dec1 = self.upconv_block(128 + 64, 64)

        # Final output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """A convolutional block with two Conv2D layers and ReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """An upsampling block with transposed convolution and a conv block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        # Decoder
        dec3 = self.dec3(torch.cat((self.upsample(bottleneck, enc3), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upsample(dec3, enc2), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upsample(dec2, enc1), enc1), dim=1))

        # Final layer
        final_output = self.final(dec1)
        return final_output

    def upsample(self, x, target):
        """Upsample `x` to the size of `target`."""
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)

# Custom Dataset to Iterate Over Multiple 4D Datasets
class Multi4DDataset(Dataset):
    def __init__(self, data_dirs):
        self.data_paths = []
        # Collect all 4D datasets from the directories
        for data_dir in data_dirs:
            for file_name in os.listdir(data_dir):
                if file_name.endswith(".pt"):  # Assume datasets are saved as .pt tensors
                    self.data_paths.append(os.path.join(data_dir, file_name))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = torch.load(self.data_paths[idx])  # Load 4D dataset
        pixel_param_data = data["pixel_param"]  # Shape: (59, 59)
        fourD_data = data["fourD"]  # Shape: (59, 59, 64, 64)
        return pixel_param_data, fourD_data

# Real-Space Pixel Dataset for Individual 4D Data
class RealSpacePixelDataset(Dataset):
    def __init__(self, pixel_param_data, fourD_data):
        self.pixel_param_data = pixel_param_data  # Shape: (59, 59)
        self.fourD_data = fourD_data  # Shape: (59, 59, 64, 64)

    def __len__(self):
        return self.fourD_data.shape[0] * self.fourD_data.shape[1]

    def __getitem__(self, idx):
        row = idx // self.fourD_data.shape[1]
        col = idx % self.fourD_data.shape[1]
        pixel_param = self.pixel_param_data[row, col]  # Scalar value
        diffraction = self.fourD_data[row, col]  # Shape: (64, 64)
        input_tensor = torch.stack([torch.full_like(diffraction, pixel_param), diffraction], dim=0)
        return input_tensor, (row, col)

# Instantiate the model
model = IntoInelastic2D()

# Paths to 4D datasets
data_dirs = ["data_dir1", "data_dir2"]  # Replace with actual paths
multi_dataset = Multi4DDataset(data_dirs)

# Split dataset into training (70%) and testing (30%)
train_size = int(0.7 * len(multi_dataset))
test_size = len(multi_dataset) - train_size
train_dataset, test_dataset = random_split(multi_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Training Loop for Multiple 4D Datasets
output_data_all = []  # Store outputs for all datasets
model.eval()

for pixel_param_data, fourD_data in train_loader:
    real_space_dataset = RealSpacePixelDataset(pixel_param_data.squeeze(0), fourD_data.squeeze(0))
    real_space_loader = DataLoader(real_space_dataset, batch_size=16, shuffle=False)
    output_data = torch.zeros(59, 59, 64, 64)

    for input_tensor, (row, col) in real_space_loader:
        with torch.no_grad():
            output = model(input_tensor)
            for i in range(len(row)):
                output_data[row[i], col[i]] = output[i].squeeze()

    output_data_all.append(output_data)
    print("Processed one training 4D dataset.")

print("Finished processing all training datasets.")

# Testing Loop
output_data_test = []
for pixel_param_data, fourD_data in test_loader:
    real_space_dataset = RealSpacePixelDataset(pixel_param_data.squeeze(0), fourD_data.squeeze(0))
    real_space_loader = DataLoader(real_space_dataset, batch_size=16, shuffle=False)
    output_data = torch.zeros(59, 59, 64, 64)

    for input_tensor, (row, col) in real_space_loader:
        with torch.no_grad():
            output = model(input_tensor)
            for i in range(len(row)):
                output_data[row[i], col[i]] = output[i].squeeze()

    output_data_test.append(output_data)
    print("Processed one test 4D dataset.")

print("Finished processing all test datasets.")
