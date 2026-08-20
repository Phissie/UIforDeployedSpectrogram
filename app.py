import os
import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# 1. Define the CNN Architecture
class DroneHelicopterCNN(nn.Module):

    def __init__(self):
        super(DroneHelicopterCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


# 2. Initialize Model and Download Weights
device = torch.device("cpu")
model = DroneHelicopterCNN()

try:
    checkpoint_path = hf_hub_download(
        repo_id="FisayoF/DeployedSpectrogram", filename="data.pkl"
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")

model.to(device)
model.eval()

# 3. Preprocessing Pipeline
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# 4. Define Inference Function
LABELS = ["Drone", "Helicopter"]


def classify_spectrogram(image):
    if image is None:
        return "Please upload a spectrogram image."

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    return {LABELS[i]: float(probabilities[i]) for i in range(len(LABELS))}


# 5. Build and Launch Web UI
demo = gr.Interface(
    fn=classify_spectrogram,
    inputs=gr.Image(type="pil", label="Upload Spectrogram Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Drone vs. Helicopter Audio Classifier",
    description="Upload an audio spectrogram image to classify whether sound originates from a Drone or a Helicopter.",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
