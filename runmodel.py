import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision

# Import model from your file
from mambaseg import LightweightEMamba

# Load model
model = LightweightEMamba(in_ch=3, out_ch=1, base_dim=48)
model.load_state_dict(torch.load("best_emmamba_core.pth", map_location='cpu'))
model.eval()

# Load and predict
img = Image.open("input.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



input_tensor = transform(img).unsqueeze(0)

torchvision.utils.save_image(input_tensor, "input_transformed.jpg")

with torch.no_grad():
    pred, _ = model(input_tensor)
    mask = (pred > 0.5).float().squeeze().numpy()

plt.imsave("polyp_segmented.png", mask, cmap='gray')
print("Saved output.png")