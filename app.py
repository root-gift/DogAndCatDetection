from flask import Flask, request, jsonify
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Chargement du modèle entraîné
model = load_model('classifier_chien_chat.keras') 

# Créer une application Flask
app = Flask(__name__)

# Définition de la route pour la prédiction
@app.route('api/predict', methods=['POST'])
def predict():
    # Obtenir le fichier d'image depuis la requête POST
    file = request.files['image']

    # Chargement et prétraiterment de l'image
    img = image.load_img(file, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Prédiction
    result = model.predict(img)

    # Interprétation des résultats de la prédiction
    if result[0][0] == 1:
        prediction = 'chien'
    else:
        prediction = 'chat'

    # Retourner la prédiction au format JSON
    return render_template("index.html",prediction = prediction)

# Lancer l'application Flask
if __name__ == '__main__':
    app.run()
