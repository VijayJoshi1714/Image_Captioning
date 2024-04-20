import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

model = resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

class_labels = {
    532: "Table",
    563: "Pen",
    767: "Pencil",
}

def classify_objects(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_indices = torch.max(outputs, 1)
        predicted_index = predicted_indices.item()
        if predicted_index in class_labels:
            predicted_label = class_labels[predicted_index]
        else:
            predicted_label = "Unknown Class"
        return predicted_label, predicted_index

image_paths = ["table.jpg", "pen.jpg", "pencil.jpg"]

for image_path in image_paths:
    predicted_label, predicted_index = classify_objects(image_path)
    print(f"Image: {image_path} - Predicted Label: {predicted_label} - Predicted Index: {predicted_index}")
