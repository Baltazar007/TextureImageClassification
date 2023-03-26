# importer les librairies nécessaires
import requests
from PIL import Image
import streamlit as st
import json
from sklearn import metrics

# définir les modèles et transformations disponibles
models = ['decision_tree', 'svm', 'catboost', 'knn']
scalers = ['original', 'standard', 'minmax']

# initialiser l'application Streamlit
st.title('Texture Image Classification')

# afficher les options de sélection du modèle et de la transformation
model = st.sidebar.selectbox('Select model:', models)
scaler = st.sidebar.selectbox('Select scaler:', scalers)

# définir le suffixe du nom de modèle en fonction de la transformation sélectionnée
if scaler == 'standard':
    scaler_suffix = '_std'
elif scaler == 'minmax':
    scaler_suffix = '_mm'
else:
    scaler_suffix = ''

# afficher le champ d'upload de l'image
image_file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])

# si une image est sélectionnée
if image_file is not None:
    # ouvrir l'image et l'afficher
    image = Image.open(image_file)
    st.image(image, caption='Uploaded image', use_column_width=True)

    # envoyer l'image au service de classification via une requête POST
    files = {'image': image_file.read()}
    response = requests.post(f'http://192.168.0.135:5000/predict', files=files)

    predictions = json.loads(response.content)
    for name, prediction in predictions.items():
        st.write(f'{name}: {prediction}')
        if isinstance(prediction, dict):
            st.write(f'Accuracy: {prediction["performance"][0]}')
            st.write(f'Precision: {prediction["performance"][1]}')
            st.write(f'Recall: {prediction["performance"][2]}')
        else:
            st.write('Unable to display performance metrics.')
