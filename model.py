
# model.py

import torch
import torchvision.transforms as transforms

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def preprocess_image(uploaded_image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
    return output.squeeze(0).squeeze(0).numpy()
