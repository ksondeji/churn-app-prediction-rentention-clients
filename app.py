import streamlit as st
import joblib
import pandas as pd

# --- Configuration de la Page ---
st.set_page_config(page_title="Prédicteur de Churn", page_icon="🎯")

# --- Chargement du Modèle (avec mise en cache) ---
@st.cache_resource
def load_model():
    # Note : Assurez-vous que 'churn_model.pkl' est dans votre repository
    try:
        model = joblib.load('churn_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Fichier du modèle non trouvé. Assurez-vous que 'churn_model.pkl' est présent.")
        return None

model = load_model()

# --- Interface Utilisateur ---
st.title("Prédiction de Churn Client")
st.write("Entrez les informations du client pour prédire son risque de départ.")

# Inputs dans la sidebar
st.sidebar.header("Paramètres du Client")
tenure = st.sidebar.slider("Ancienneté (mois)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Facture mensuelle (€)", 20, 120, 50)

# Bouton de prédiction
if st.button("Lancer la Prédiction", type="primary"):
    if model is not None:
        input_data = pd.DataFrame([[tenure, monthly_charges]], columns=['tenure', 'MonthlyCharges'])

        with st.spinner("Prédiction en cours..."):
            prediction = model.predict_proba(input_data)[0][1]

        st.subheader("Résultat de la Prédiction")
        if prediction > 0.5:
            st.error(f"Risque de Churn Élevé : {prediction:.1%}")
        else:
            st.success(f"Risque de Churn Faible : {prediction:.1%}")
    else:
        st.warning("Le modèle n'a pas pu être chargé.")

# --- Poussez les changements sur GitHub, et Streamlit mettra à jour l'app automatiquement ! ---
