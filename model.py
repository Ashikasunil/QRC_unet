# model.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel):
        super(MobileViTBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, dim, kernel_size=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim*4),
            num_layers=depth
        )
        self.conv3 = nn.Conv2d(dim, channel, kernel_size=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        x = self.conv3(x)
        return x + res

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class QRC_UNet(nn.Module):
    def __init__(self):
        super(QRC_UNet, self).__init__()
        self.encoder1 = MobileViTBlock(dim=64, depth=2, channel=3)
        self.encoder2 = MobileViTBlock(dim=128, depth=2, channel=64)
        self.encoder3 = MobileViTBlock(dim=256, depth=2, channel=128)
        self.encoder4 = MobileViTBlock(dim=512, depth=2, channel=256)
        self.bottleneck = MobileViTBlock(dim=1024, depth=2, channel=512)
        self.decoder4 = DecoderBlock(1024, 512, 512)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        x = self.bottleneck(skip4)
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.final_conv(x)
        return x

def load_model(model_path):
    model = QRC_UNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(uploaded_image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image)
    image = image.unsqueeze(0)
    return image

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
    return output.squeeze(0).squeeze(0).numpy()
