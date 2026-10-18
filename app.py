from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np

# ==========================================
# 1. INITIALISATION DE L'API
# ==========================================
app = FastAPI(
    title="API Diagnostic IRM", 
    description="API MLOps de classification des tumeurs cérébrales (ResNet18)",
    version="1.0.0"
)

# ==========================================
# 2. CHARGEMENT GLOBAL DU MODÈLE (Au démarrage)
# Règle d'ingénierie : Ne jamais charger le modèle à l'intérieur de la route web.
# ==========================================
print("Démarrage du moteur d'inférence...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('meilleur_modele_resnet_cropped.pth', map_location=device))
model = model.to(device)
model.eval()
print("Modèle chargé en mémoire avec succès.")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. PRÉTRAITEMENT OPENCV ADAPTÉ AU WEB (Bytes)
# ==========================================
def auto_crop_mri_bytes(image_bytes):
    """Applique le recadrage directement sur un flux d'octets sans écrire sur le disque."""
    # Conversion du flux web en matrice OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Le fichier reçu n'est pas une image valide.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped_img = img[y:y+h, x:x+w]
    else:
        cropped_img = img
        
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_img_rgb)


# ==========================================
# 4. LE POINT D'ENTRÉE CLINIQUE (Endpoint POST)
# ==========================================
@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    """Route API qui reçoit une image IRM et retourne le diagnostic en JSON."""
    try:
        # 1. Lecture du fichier uploadé
        image_bytes = await file.read()
        
        # 2. Extraction chirurgicale OpenCV
        img_pil = auto_crop_mri_bytes(image_bytes)
        
        # 3. Préparation Tensor
        img_tensor = inference_transform(img_pil).unsqueeze(0).to(device)
        
        # 4. Inférence mathématique
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0] * 100
            
        confiance_max, predicted_idx = torch.max(probabilities, 0)
        classe_predite = classes[predicted_idx.item()]
        
        # 5. Réponse JSON formatée pour l'application cliente
        return {
            "statut": "succès",
            "fichier": file.filename,
            "diagnostic": classe_predite,
            "confiance_globale": round(confiance_max.item(), 2),
            "probabilites_detaillees": {
                classes[i]: round(probabilities[i].item(), 2) for i in range(4)
            }
        }
        
    except Exception as e:
        # Gestion stricte des erreurs (ex: fichier non image)
        raise HTTPException(status_code=400, detail=str(e))

# ==========================================
# 5. DÉMARRAGE DU SERVEUR
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)