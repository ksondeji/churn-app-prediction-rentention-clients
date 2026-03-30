import streamlit as st

st.set_page_config(page_title="Mon App Incroyable", page_icon="🚀")

st.title("Mon Premier Déploiement !")

st.write("Félicitations, votre application est en ligne !")

nom = st.text_input("Quel est votre nom ?")

if st.button("Dire bonjour"):
    if nom:
        st.success(f"Bonjour, {nom} !")
    else:
        st.warning("Veuillez entrer un nom.")
