"""
Gram Stain Classifier — Prediction Module
Model    : EfficientNet-B0
Accuracy : 98% on test set
Classes  : gram_negative (0), gram_positive (1)

How to use:
    from predict import load_model, predict_gram
    model, class_names = load_model("gram_classifier.pth")
    result = predict_gram("image.jpg", model, class_names)
"""

import torch
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from pathlib import Path


# ==============================
# ✅ Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# ✅ Supported Formats
# ==============================
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}


# ==============================
# ✅ Load Model
# ==============================
def load_model(model_path: str):
    """
    Load saved EfficientNet model from .pth file

    Args:
        model_path : path to gram_classifier.pth

    Returns:
        model       : loaded model ready for inference
        class_names : ["gram_negative", "gram_positive"]
    """
    checkpoint = torch.load(model_path, map_location=device)

    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint["class_names"]


# ==============================
# ✅ Gram Stain Validator
# ==============================
def is_gram_stain_image(image_path: str) -> tuple:
    """
    Checks if image looks like a real Gram stain microscope image.

    Strategy:
    Instead of checking exact color values (which vary a lot between
    different microscopes and staining batches), we focus on what ALL
    wrong images have in common:
    - Too dark  (photos taken in low light)
    - No color variation (blank images, solid colors)
    - Green dominant (natural/outdoor photos, green backgrounds)
    - Human skin tone (selfies) → R >> G >> B balanced warm tone

    We intentionally keep checks LOOSE to avoid rejecting valid
    gram stain images that have slightly different backgrounds.

    Args:
        image_path : path to image file

    Returns:
        (True, "ok")         → valid gram stain image
        (False, reason: str) → not a valid gram stain image
    """
    img    = Image.open(image_path).convert("RGB")
    np_img = np.array(img).astype(float)

    mean_r = np_img[:, :, 0].mean()
    mean_g = np_img[:, :, 1].mean()
    mean_b = np_img[:, :, 2].mean()

    # ── Check 1 — Not too dark ──
    # Gram stain images always have a bright white/cream background
    # Threshold kept low (50) to handle dim microscope photos
    overall_brightness = np_img.mean()
    if overall_brightness < 50:
        return False, "Image is too dark to be a Gram stain"

    # ── Check 2 — Has color variation ──
    # A blank, solid color, or near-blank image fails this
    # Threshold kept low (8) to handle pale/lightly stained images
    color_std = np_img.std()
    if color_std < 8:
        return False, "Image has no color variation — not a valid stain image"

    # ── Check 3 — Reject clearly green dominant images ──
    # Natural/outdoor photos have green as the highest channel
    # Gram stains NEVER have green as dominant channel
    if mean_g > mean_r and mean_g > mean_b:
        return False, "Image appears to be a natural or outdoor photo"

    # ── Check 4 — Reject human skin tone ──
    # Skin tone: R is clearly highest, G is second, B is lowest
    # AND the difference between R and B is large (warm tone)
    # AND green is higher than blue (typical skin)
    skin_tone = (
        mean_r > mean_g > mean_b and   # R > G > B ordering
        (mean_r - mean_b) > 40 and     # large warm gap
        mean_g > 100                    # not dark skin (already caught by check 1)
    )
    if skin_tone:
        return False, "Image appears to be a photo of a person"

    # ── Check 5 — Reject very uniform images (solid backgrounds) ──
    # Checks each channel separately for variation
    r_std = np_img[:, :, 0].std()
    g_std = np_img[:, :, 1].std()
    b_std = np_img[:, :, 2].std()

    if r_std < 5 and g_std < 5 and b_std < 5:
        return False, "Image appears to be blank or a solid color"

    return True, "ok"


# ==============================
# ✅ Predict Function
# ==============================
def predict_gram(image_path: str, model, class_names: list, threshold: float = 0.95):
    """
    Predict Gram stain classification for a single image.

    Args:
        image_path  : path to image file
        model       : loaded model from load_model()
        class_names : list of class names
        threshold   : confidence threshold (default 0.95 = 95%)

    Supported formats:
        .jpg .jpeg .png .bmp .tiff .tif .webp .gif

    Returns dict:
        prediction   : "gram_positive" or "gram_negative" or None
        confidence   : float (0-100) percentage
        all_probs    : dict of all class probabilities
        is_confident : True if prediction is reliable and confident
        warning      : warning message if something is wrong else None
    """

    # ── Step 1 — Format Check ──
    ext = Path(image_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: '{ext}'\n"
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    # ── Step 2 — Gram Stain Validation ──
    is_valid, reason = is_gram_stain_image(image_path)
    if not is_valid:
        return {
            "prediction":   None,
            "confidence":   0.0,
            "all_probs":    {},
            "is_confident": False,
            "warning":      f"Image does not appear to be a Gram stain: {reason}"
        }

    # ── Step 3 — Preprocessing ──
    # Must match exactly what was used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # .convert("RGB") handles: RGBA, grayscale, palette mode automatically
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # ── Step 4 — Inference ──
    with torch.no_grad():
        outputs              = model(tensor)
        probs                = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    confidence = confidence.item()
    pred_idx   = pred_idx.item()
    prediction = class_names[pred_idx]

    # ── Step 5 — All Class Probabilities ──
    all_probs = {
        class_names[i]: round(probs[0][i].item() * 100, 2)
        for i in range(len(class_names))
    }

    # ── Step 6 — Reliability Check ──
    # If margin between classes < 30% model is uncertain
    prob_values  = list(all_probs.values())
    margin       = abs(prob_values[0] - prob_values[1])
    is_reliable  = margin >= 30.0

    # ── Step 7 — Build Warning Message ──
    is_confident = confidence >= threshold and is_reliable

    if not is_reliable:
        warning = (
            "Prediction is unreliable — model is uncertain between classes. "
            "Please upload a clearer Gram stain image."
        )
    elif not is_confident:
        warning = (
            f"Low confidence ({confidence * 100:.1f}%). "
            f"Please perform confirmatory biochemical tests."
        )
    else:
        warning = None

    return {
        "prediction":   prediction,
        "confidence":   round(confidence * 100, 2),
        "all_probs":    all_probs,
        "is_confident": is_confident,
        "warning":      warning
    }


# ==============================
# ✅ FastAPI Integration
# ==============================
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from predict import load_model, predict_gram
import shutil
import os

app                = FastAPI()
model, class_names = load_model("gram_classifier.pth")


@app.post("/predict-gram")
async def predict(file: UploadFile = File(...)):

    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {'.jpg', '.jpeg', '.png', '.bmp',
                   '.tiff', '.tif', '.webp', '.gif'}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}"
        )

    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict_gram(temp_path, model, class_names)
    finally:
        os.remove(temp_path)   # always clean up

    return result


# ── Response Examples ──

# Valid Gram stain image:
# {
#     "prediction":   "gram_negative",
#     "confidence":   99.9,
#     "all_probs":    {"gram_negative": 99.9, "gram_positive": 0.1},
#     "is_confident": true,
#     "warning":      null
# }

# Low confidence:
# {
#     "prediction":   "gram_positive",
#     "confidence":   75.3,
#     "all_probs":    {"gram_negative": 24.7, "gram_positive": 75.3},
#     "is_confident": false,
#     "warning":      "Low confidence (75.3%). Please perform confirmatory biochemical tests."
# }

# Not a Gram stain image (e.g. selfie):
# {
#     "prediction":   null,
#     "confidence":   0.0,
#     "all_probs":    {},
#     "is_confident": false,
#     "warning":      "Image does not appear to be a Gram stain: Image appears to be a photo of a person"
# }

# Unsupported format:
# HTTP 400: "Unsupported file format: .pdf"
"""
