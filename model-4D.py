import torch
import torch.nn as nn
import torch.nn.functional as F

class IntoInelastic(nn.Module):
    def __init__(self):
        super(IntoInelastic, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(2, 64) # 2 input channels (1)+(2), to 64 channels
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.upconv_block(1024 + 512, 512)
        self.dec3 = self.upconv_block(512 + 256, 256)
        self.dec2 = self.upconv_block(256 + 128, 128)
        self.dec1 = self.upconv_block(128 + 64, 64)

        # Final output layer
        self.final = nn.Conv3d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """A convolutional block with two Conv3D layers and ReLU activations."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """An upsampling block with transposed convolution and a conv block."""
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Reshape input to flatten real-space dimensions
        batch_size, channels, real1, real2, diff1, diff2 = x.shape
        x = x.view(batch_size, channels, real1 * real2, diff1, diff2)  # Flatten real-space dimensions

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, (2, 2, 2)))  # Pool across flattened real-space and diffraction space
        enc3 = self.enc3(F.max_pool3d(enc2, (2, 2, 2)))
        enc4 = self.enc4(F.max_pool3d(enc3, (2, 2, 2)))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, (2, 2, 2)))

        # Decoder
        dec4 = self.dec4(torch.cat((self.upsample(bottleneck, enc4), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upsample(dec4, enc3), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upsample(dec3, enc2), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upsample(dec2, enc1), enc1), dim=1))

        # Final layer
        final_output = self.final(dec1)

        # Reshape output to restore real-space dimensions
        final_output = final_output.view(batch_size, 1, real1, real2, diff1, diff2)
        return final_output

    def upsample(self, x, target):
        """Upsample `x` to the size of `target`."""
        return F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=True)
        
# Instantiate the model
model = IntoInelastic()

# Example input tensor
# Input has 1 channel for the (1) "elastic" image  and 1 channel for (2) atomic numbers (z), so total input channels = 2
# Real space (59, 59) and diffraction space (64, 64)
input_tensor = torch.randn(1, 2, 59, 59, 64, 64)  # Batch size = 1
output = model(input_tensor)

print("Output shape:", output.shape)  # Should match input spatial dimensions (1, 1, 59, 59, 64, 64)
