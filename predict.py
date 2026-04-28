import torch
import timm
from torchvision import transforms
from PIL import Image

# Load model once when server starts
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = timm.create_model('resnet50', pretrained=False, num_classes=3)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

CLASS_NAMES = ['gram_negative', 'gram_positive', 'not_gram_stain']

def predict(image_path, model, threshold=0.60):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        probs     = torch.softmax(model(tensor), dim=1)[0]
        conf, idx = torch.max(probs, 0)

    conf       = conf.item()
    prediction = CLASS_NAMES[idx.item()]
    all_probs  = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
                  for i in range(3)}

    # Case 1 — too uncertain to trust at all
    if conf < threshold:
        return {
            'prediction':  'uncertain',
            'confidence':  round(conf * 100, 2),
            'all_probs':   all_probs,
            'status':      'rejected',
            'message':     'Image does not appear to be a valid gram-stained slide. Please upload a clearer microscope image.'
        }

    # Case 2 — confident enough but below 90% — warn user
    if conf < 0.90:
        return {
            'prediction':  prediction,
            'confidence':  round(conf * 100, 2),
            'all_probs':   all_probs,
            'status':      'low_confidence',
            'message':     f'Prediction is {prediction} but please perform confirmatory biochemical tests to verify this result.'
        }

    # Case 3 — high confidence result
    return {
        'prediction':  prediction,
        'confidence':  round(conf * 100, 2),
        'all_probs':   all_probs,
        'status':      'high_confidence',
        'message':     f'Proceed with {prediction} diagnostic pathway.'
                       if prediction != 'not_gram_stain'
                       else 'Image does not appear to be a gram-stained slide.'
    }