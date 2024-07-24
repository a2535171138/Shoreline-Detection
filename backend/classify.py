import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image


def classify_image(image, model_path='coast_classifier.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    def preprocess_image(image):
        image = image.convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        return image

    def predict_image(image):
        image = preprocess_image(image)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            return predicted.item()

    return predict_image(image)


# # example
# image = Image.open('Places365/c/canal/natural/00000189.jpg')
# result = classify_image(image)
