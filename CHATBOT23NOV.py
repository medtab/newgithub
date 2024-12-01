import os
import pandas as pd
import spacy
import numpy as np
import streamlit as st
import re
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Définir les chemins des fichiers
script_path = os.path.abspath(__file__)
base_path = os.path.dirname(script_path)

# Chemins des fichiers
csv_path = os.path.join(base_path, r"C:\pycharm\PyCharm Community Edition 2024.2.4\bin\conda\pythonProject\CHATBOT23NOV.csv")
logo_path = os.path.join(base_path, r"C:\pycharm\PyCharm Community Edition 2024.2.4\bin\conda\pythonProject\Logo_21 - 23NOV.png")
response_image_path = os.path.join(base_path, r"C:\pycharm\PyCharm Community Edition 2024.2.4\bin\conda\pythonProject\IMG_23NOV.jpg")

# Chargement des modèles
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("fr_core_news_sm")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle SpaCy : {e}")
        return None

@st.cache_resource
def load_sentence_transformer_model():
    try:
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle SentenceTransformer : {e}")
        return None

@st.cache_data
def load_faq_data(file_path):
    if not os.path.isfile(file_path):
        st.error(f"Erreur : le fichier '{file_path}' n'existe pas.")
        return None

    try:
        faq_data = pd.read_csv(file_path, sep=';', encoding='latin-1', on_bad_lines='warn')
        faq_data.columns = faq_data.columns.str.strip()
        if {'Questions', 'Answers'}.issubset(faq_data.columns):
            st.success("Bienvenue, comment puis-je vous aider ?")
            return faq_data
        else:
            st.error("Erreur : le fichier doit contenir les colonnes 'Questions' et 'Answers'.")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier FAQ : {e}")
        return None

def preprocess_question(question):
    if nlp is None:
        st.error("Le modèle SpaCy n'a pas pu être chargé, impossible de prétraiter la question.")
        return None
    doc = nlp(question.lower())
    corrected = " ".join(token.lemma_ for token in doc if not token.is_stop)
    return re.sub(r'\W+', ' ', corrected).strip()

def calculate_faq_embeddings(faq_data):
    return model.encode(faq_data['Questions'].fillna('').tolist(), convert_to_tensor=True)

def find_similar_question(question, faq_embeddings):
    question_embedding = model.encode([question], convert_to_tensor=True)
    similarities = cosine_similarity(question_embedding, faq_embeddings)
    top_index = np.argmax(similarities)
    return top_index, similarities[0][top_index]

def refine_response(index, similarity_score, faq_data, threshold=0.7):
    if similarity_score > threshold:
        return faq_data['Answers'].iloc[index]
    else:
        return "Désolé, je n'ai pas trouvé de réponse à votre question."

def chatbot_pipeline(question, faq_data, faq_embeddings):
    processed_question = preprocess_question(question)
    if processed_question is None:
        return "Erreur lors du prétraitement de la question."
    index, similarity_score = find_similar_question(processed_question, faq_embeddings)
    return refine_response(index, similarity_score, faq_data)

def main():
    # Application du style CSS
    st.markdown("""
    <style>
        .title {
            color: green;
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Créer des colonnes pour le logo et le titre
    col1, col2 = st.columns([1, 3])

    # Afficher le logo dans la première colonne
    if os.path.isfile(logo_path):
        logo_image = Image.open(logo_path).resize((100, 100))
        col1.image(logo_image)
    else:
        col1.warning(f"Logo introuvable à l'emplacement : {logo_path}")

    # Afficher le titre dans la deuxième colonne
    with col2:
        st.markdown('<div class="title">Chatbot de la Société d\'Environnement et de Plantation de Redeyef (SEPR)</div>', unsafe_allow_html=True)

    # Charger les modèles
    global nlp, model
    nlp = load_spacy_model()
    if nlp is None:
        st.stop()  # Arrête l'exécution si le modèle ne charge pas

    model = load_sentence_transformer_model()
    if model is None:
        st.stop()  # Arrête l'exécution si le modèle ne charge pas

    # Charger les données FAQ
    faq_data = load_faq_data(csv_path)
    if faq_data is not None:
        faq_embeddings = calculate_faq_embeddings(faq_data)

        # Saisie de la question par l'utilisateur
        user_question = st.text_input("Posez votre question ici :", "")
        if st.button("Obtenir une réponse") and user_question:
            response = chatbot_pipeline(user_question, faq_data, faq_embeddings)
            st.markdown(f"**Réponse :** {response}")

            # Afficher une image associée à la réponse
            if os.path.isfile(response_image_path):
                response_image = Image.open(response_image_path).resize((400, 300))
                st.image(response_image, caption="Illustration associée")
            else:
                st.warning(f"Image de réponse introuvable à l'emplacement : {response_image_path}")

if __name__ == "__main__":
    main()