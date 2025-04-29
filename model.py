
# model.py

import torch
import torchvision.transforms as transforms

def load_model(model_path):
    loaded = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(loaded, dict):  # state_dict
        from model_arch import QRC_UNet
        model = QRC_UNet()
        model.load_state_dict(loaded)
    else:  # full model
        model = loaded
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
