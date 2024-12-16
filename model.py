import torch
import torch.nn as nn
import torch.nn.functional as F

class IntoInelastic(nn.Module):
    def __init__(self):
        super(IntoInelastic, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(2, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)

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
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.dec4(torch.cat((self.upsample(bottleneck, enc4), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upsample(dec4, enc3), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upsample(dec3, enc2), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upsample(dec2, enc1), enc1), dim=1))

        # Final layer
        return self.final(dec1)

    def upsample(self, x, target):
        """Upsample `x` to the size of `target`."""
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)

# Instantiate the model
model = IntoInelastic()

# Example input tensor
# Input has 1 channel for the (1) "elastic" image  and 1 channel for (2) atomic numbers (z), so total input channels = 2
input_tensor = torch.randn(1, 2, 256, 256)  # Batch size = 1, H = W = 256
output = model(input_tensor)

print("Output shape:", output.shape)  # Should be [1, 1, 256, 256]
