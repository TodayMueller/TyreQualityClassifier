import os, base64
from PIL import Image, ImageEnhance, ImageStat
from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from nets.detect_model import DetectionNet
from nets.class_model   import ClassificationNet

detection_model = DetectionNet().to(device)
detection_model.load_state_dict(torch.load("nets/detection_net_state.pth", map_location=device))
detection_model.eval()

classification_model = ClassificationNet.to(device)
classification_model.load_state_dict(torch.load("nets/class_net_state.pth", map_location=device))
classification_model.eval()

ALLOWED_EXTENSIONS = {"jpg","jpeg","png"}

def allowed_file(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def process_tensor(path):
    size = os.path.getsize(path)
    if size < 61*1024 or size > 10*1024*1024:
        raise ValueError("error")
    img = Image.open(path)
    w,h = img.size
    if w<282 or h<282:
        raise ValueError("error")
    gray = img.convert("L")
    sd = ImageStat.Stat(gray).stddev[0]
    if 0<sd<24:
        img = ImageEnhance.Contrast(img).enhance(24/sd)
    tf = transforms.Compose([
        transforms.Resize((336,336)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    return tf(img).unsqueeze(0).to(device)

def classify_image(path):
    tensor = process_tensor(path)
    with torch.no_grad():
        det = torch.sigmoid(detection_model(tensor)).item() > 0.5
        if not det:
            return "external", None
        prob = float(torch.sigmoid(classification_model(tensor)).item())
        return ("good" if prob>0.5 else "defective"), prob

def encode_image_to_base64(path):
    with open(path,'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')
