# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib as jb
# import plotly.graph_objects as go
# import plotly.express as px

# # ============================================================
# # CONFIGURATION GÉNÉRALE DE LA PAGE
# # ============================================================
# st.set_page_config(
#     page_title="Marketing Bancaire | Prédiction de souscription",
#     page_icon="🏦",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ============================================================
# # THÈME PERSONNALISÉ — BLEU FONCÉ / BLEU CLAIR
# # ============================================================
# PRIMARY_DARK = "#0B2447"     # bleu foncé principal
# PRIMARY_MID = "#19376D"      # bleu foncé intermédiaire
# ACCENT_BLUE = "#576CBC"      # bleu moyen (accents, boutons)
# LIGHT_BLUE = "#A5D7E8"       # bleu clair (fonds, cartes)
# TEXT_LIGHT = "#FFFFFF"       # texte clair sur fond foncé
# TEXT_DARK = "#0B2447"        # texte foncé sur fond clair

# st.markdown(f"""
# <style>
#     /* Fond général de l'application */
#     .stApp {{
#         # background: linear-gradient(180deg, {PRIMARY_DARK} 0%, {PRIMARY_MID} 100%);
#         color: {TEXT_LIGHT};
#     }}

#     /* Titres */
#     h1, h2, h3 {{
#         color: {TEXT_LIGHT} !important;
#         font-weight: 800 !important;
#     }}

#     /* Barre latérale */
#     section[data-testid="stSidebar"] {{
#         background: linear-gradient(180deg, {PRIMARY_DARK} 0%, {PRIMARY_MID} 100%);
#         border-right: 2px solid {ACCENT_BLUE};
#     }}
#     section[data-testid="stSidebar"] * {{
#         color: {TEXT_LIGHT} !important;
#     }}

#     /* Cartes / conteneurs */
#     div[data-testid="stVerticalBlockBorderWrapper"] {{
#         background-color: {LIGHT_BLUE}15;
#         border: 1px solid {ACCENT_BLUE}55;
#         border-radius: 16px;
#         padding: 6px;
#     }}

#     /* Onglets */
#     button[data-baseweb="tab"] {{
#         background-color: #FFFFFF;
#         color: #FFFFFF !important;
#         border-radius: 10px 10px 0 0;
#         font-weight: 600;
#         padding: 10px 20px;
#     }}
#     button[data-baseweb="tab"][aria-selected="true"] {{
#         background-color: {ACCENT_BLUE};
#         color: white !important;
#     }}

#     /* Boutons */
#     .stButton>button {{
#         background: linear-gradient(90deg, {ACCENT_BLUE}, {LIGHT_BLUE});
#         color: {TEXT_DARK};
#         font-weight: 700;
#         border: none;
#         border-radius: 10px;
#         padding: 0.6em 1.5em;
#         transition: 0.2s;
#     }}
#     .stButton>button:hover {{
#         transform: scale(1.03);
#         box-shadow: 0 0 12px {LIGHT_BLUE}AA;
#     }}

#     /* Champs de saisie */
#     .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
#         background-color: {TEXT_LIGHT} !important;
#         color: {TEXT_LIGHT} !important;
#         border-radius: 8px !important;
#         font-weight: 500;
#     }}

#     /* Métriques */
#     div[data-testid="stMetric"] {{
#         background-color: {LIGHT_BLUE}22;
#         border: 1px solid {ACCENT_BLUE}66;
#         border-radius: 14px;
#         padding: 14px;
#     }}
#     div[data-testid="stMetricValue"] {{
#         color: {LIGHT_BLUE} !important;
#     }}

#     /* Bannière résultat */
#     .result-banner {{
#         border-radius: 16px;
#         padding: 22px;
#         text-align: center;
#         font-size: 1.4rem;
#         font-weight: 800;
#         margin-top: 10px;
#     }}
#     .result-oui {{
#         background: linear-gradient(90deg, #1B998B, #A5D7E8);
#         color: {TEXT_DARK};
#     }}
#     .result-non {{
#         background: linear-gradient(90deg, #576CBC, #19376D);
#         color: {TEXT_LIGHT};
#     }}

#     /* Séparateurs */
#     hr {{
#         border-color: {ACCENT_BLUE}55;
#     }}
# </style>
# """, unsafe_allow_html=True)

# # ============================================================
# # CHARGEMENT DES RESSOURCES (mise en cache)
# # ============================================================
# COLONNES = ['age', 'job', 'marital', 'education', 'housing', 'loan',
#             'contact', 'month', 'day_of_week', 'duration', 'campaign',
#             'pdays', 'previous', 'poutcome']

# CAT_INDEX = {
#     'job': 0, 'marital': 1, 'education': 2, 'housing': 3, 'loan': 4,
#     'contact': 5, 'month': 6, 'day_of_week': 7, 'poutcome': 8,
# }


# @st.cache_resource(show_spinner="Chargement du modèle et des encodeurs...")
# def load_resources():
#     encoders = jb.load('encodes.joblib')
#     cat_goriell = jb.load('cat_goriell.joblib')
#     scaler = jb.load('scaler.joblib')
#     gb = jb.load('gb_model.joblib')
#     clasnames = cat_goriell[9]
#     return encoders, cat_goriell, scaler, gb, clasnames


# try:
#     encoders, cat_goriell, scaler, gb, clasnames = load_resources()
#     RESOURCES_OK = True
# except Exception as e:
#     RESOURCES_OK = False
#     LOAD_ERROR = str(e)


# # ============================================================
# # FONCTION DE PRÉDICTION
# # ============================================================
# def build_feature_vector(age, job, marital, education, housing, loan,
#                           contact, month, day_of_week, duration, campaign,
#                           pdays, previous, poutcome):
#     job_enc = encoders[0].transform([job])[0]
#     marital_enc = encoders[1].transform([marital])[0]
#     education_enc = encoders[2].transform([education])[0]
#     housing_enc = encoders[3].transform([housing])[0]
#     loan_enc = encoders[4].transform([loan])[0]
#     contact_enc = encoders[5].transform([contact])[0]
#     month_enc = encoders[6].transform([month])[0]
#     day_of_week_enc = encoders[7].transform([day_of_week])[0]
#     poutcome_enc = encoders[8].transform([poutcome])[0]

#     x_new = np.array([age, job_enc, marital_enc, education_enc,
#                        housing_enc, loan_enc, contact_enc, month_enc,
#                        day_of_week_enc, duration, campaign, pdays,
#                        previous, poutcome_enc], dtype=float).reshape(1, -1)
#     return scaler.transform(x_new)


# def Pred_func(age, job, marital, education, housing, loan, contact, month,
#               day_of_week, duration, campaign, pdays, previous, poutcome,
#               return_proba=False):
#     x_new = build_feature_vector(age, job, marital, education, housing, loan,
#                                   contact, month, day_of_week, duration,
#                                   campaign, pdays, previous, poutcome)
#     y_pred = gb.predict(x_new)[0]
#     label = clasnames[y_pred]

#     if return_proba and hasattr(gb, "predict_proba"):
#         proba = gb.predict_proba(x_new)[0]
#         return label, proba
#     return label, None


# def Pred_func_csv(df):
#     predictions, probas_oui = [], []
#     has_proba = hasattr(gb, "predict_proba")
#     for row in df.itertuples(index=False):
#         try:
#             label, proba = Pred_func(
#                 row.age, row.job, row.marital, row.education, row.housing,
#                 row.loan, row.contact, row.month, row.day_of_week,
#                 row.duration, row.campaign, row.pdays, row.previous,
#                 row.poutcome, return_proba=has_proba
#             )
#             predictions.append(label)
#             probas_oui.append(round(float(max(proba)) * 100, 1) if proba is not None else None)
#         except Exception as e:
#             predictions.append(f"Erreur: {e}")
#             probas_oui.append(None)

#     df_out = df.copy()
#     df_out['prediction'] = predictions
#     if has_proba:
#         df_out['confiance_%'] = probas_oui
#     return df_out


# # ============================================================
# # EN-TÊTE
# # ============================================================
# st.markdown(f"""
# <div style="text-align:center; padding: 10px 0 20px 0;">
#     <h1>🏦 Marketing Bancaire — Prédiction des souscriptions</h1>
#     <p style="color:{LIGHT_BLUE}; font-size:1.05rem;">
#         Anticipez, avec un modèle Gradient Boosting, si un client souscrira à un
#         dépôt à terme — et optimisez l'impact de vos campagnes.
#     </p>
# </div>
# <hr>
# """, unsafe_allow_html=True)

# if not RESOURCES_OK:
#     st.error(
#         "Impossible de charger les fichiers du modèle "
#         "(`encodes.joblib`, `cat_goriell.joblib`, `scaler.joblib`, `gb_model.joblib`). "
#         f"Détail : {LOAD_ERROR}\n\n"
#         "Placez ces fichiers dans le même dossier que `app.py` avant de relancer."
#     )
#     st.stop()

# # ============================================================
# # BARRE LATÉRALE — INFOS & STATISTIQUES
# # ============================================================
# with st.sidebar:
#     st.markdown("## ℹ️ À propos")
#     st.write(
#         "Ce tableau de bord utilise un modèle de machine learning basé sur **Gradient Boosting** entraîné "
#         "sur des données de campagnes marketing bancaires pour prédire la "
#         "souscription à un dépôt à terme."
#     )
#     st.markdown("### 🎯 Classes cible")
#     for c in clasnames:
#         st.markdown(f"- **{c}**")
#     st.markdown("---")
#     st.markdown("### 🧩 Variables utilisées")
#     st.write(", ".join(COLONNES))
#     st.markdown("---")
#     st.caption("Développé avec Streamlit • Thème bleu bancaire")

# # ============================================================
# # ONGLETS PRINCIPAUX
# # ============================================================
# tab1, tab2 = st.tabs(["🧑‍💼 Prédiction simple", "📂 Prédiction multiple (CSV)"])

# # ------------------------------------------------------------
# # ONGLET 1 : PRÉDICTION SIMPLE
# # ------------------------------------------------------------
# with tab1:
#     col_form, col_result = st.columns([1.3, 1])

#     with col_form:
#         st.subheader("Informations sur le client")

#         c1, c2 = st.columns(2)
#         with c1:
#             age = st.number_input("Âge", min_value=17, max_value=100, value=35)
#             job = st.selectbox("Emploi", cat_goriell[CAT_INDEX['job']])
#             marital = st.selectbox("Statut matrimonial", cat_goriell[CAT_INDEX['marital']])
#             education = st.selectbox("Éducation", cat_goriell[CAT_INDEX['education']])
#             housing = st.selectbox("Prêt immobilier", cat_goriell[CAT_INDEX['housing']])
#             loan = st.selectbox("Prêt personnel", cat_goriell[CAT_INDEX['loan']])
#             contact = st.selectbox("Type de contact", cat_goriell[CAT_INDEX['contact']])

#         with c2:
#             month = st.selectbox("Mois du dernier contact", cat_goriell[CAT_INDEX['month']])
#             day_of_week = st.selectbox("Jour de la semaine", cat_goriell[CAT_INDEX['day_of_week']])
#             duration = st.number_input("Durée du dernier appel (s)", min_value=0, value=180)
#             campaign = st.number_input("Nb de contacts (campagne)", min_value=1, value=2)
#             pdays = st.number_input("Jours depuis dernier contact (999 = jamais)", min_value=0, value=999)
#             previous = st.number_input("Contacts avant cette campagne", min_value=0, value=0)
#             poutcome = st.selectbox("Résultat campagne précédente", cat_goriell[CAT_INDEX['poutcome']])

#         predict_btn = st.button("🔍 Lancer la prédiction", use_container_width=True)

#     with col_result:
#         st.subheader("Résultat")
#         if predict_btn:
#             try:
#                 label, proba = Pred_func(
#                     age, job, marital, education, housing, loan, contact,
#                     month, day_of_week, duration, campaign, pdays, previous,
#                     poutcome, return_proba=True
#                 )

#                 is_yes = str(label).lower() in ("yes", "oui", "1", "true")
#                 css_class = "result-oui" if is_yes else "result-non"
#                 icon = "✅" if is_yes else "❌"

#                 st.markdown(
#                     f"""<div class="result-banner {css_class}">
#                         {icon} Souscription prédite : <u>{label}</u>
#                     </div>""",
#                     unsafe_allow_html=True
#                 )

#                 if proba is not None:
#                     confiance = round(float(max(proba)) * 100, 1)
#                     st.metric("Confiance du modèle", f"{confiance} %")

#                     fig = go.Figure(go.Indicator(
#                         mode="gauge+number",
#                         value=confiance,
#                         number={'suffix': " %", 'font': {'color': TEXT_LIGHT}},
#                         gauge={
#                             'axis': {'range': [0, 100], 'tickcolor': TEXT_LIGHT},
#                             'bar': {'color': LIGHT_BLUE},
#                             'bgcolor': PRIMARY_MID,
#                             'bordercolor': ACCENT_BLUE,
#                             'steps': [
#                                 {'range': [0, 50], 'color': PRIMARY_DARK},
#                                 {'range': [50, 80], 'color': PRIMARY_MID},
#                                 {'range': [80, 100], 'color': ACCENT_BLUE},
#                             ],
#                         },
#                         title={'text': "Niveau de confiance", 'font': {'color': TEXT_LIGHT}},
#                     ))
#                     fig.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         font={'color': TEXT_LIGHT},
#                         height=280,
#                         margin=dict(l=20, r=20, t=50, b=10),
#                     )
#                     st.plotly_chart(fig, use_container_width=True)

#             except Exception as e:
#                 st.error(f"Erreur lors de la prédiction : {e}")
#         else:
#             st.info("Renseignez les informations du client puis cliquez sur "
#                     "**Lancer la prédiction**.")

# # ------------------------------------------------------------
# # ONGLET 2 : PRÉDICTION MULTIPLE (CSV)
# # ------------------------------------------------------------
# with tab2:
#     st.subheader("Prédiction par lot (fichier CSV)")
#     st.write(
#         f"Le fichier doit contenir exactement les colonnes suivantes, dans "
#         f"n'importe quel ordre : `{', '.join(COLONNES)}`"
#     )

#     uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             missing = [c for c in COLONNES if c not in df.columns]

#             if missing:
#                 st.error(f"Colonnes manquantes dans le fichier : {missing}")
#             else:
#                 st.success(f"Fichier chargé avec succès — {len(df)} lignes détectées.")
#                 st.dataframe(df.head(), use_container_width=True)

#                 if st.button("🚀 Lancer les prédictions", use_container_width=True):
#                     with st.spinner("Prédiction en cours..."):
#                         df_result = Pred_func_csv(df)

#                     st.markdown("### 📊 Résultats")
#                     st.dataframe(df_result, use_container_width=True)

#                     # Visualisation synthèse
#                     counts = df_result['prediction'].value_counts().reset_index()
#                     counts.columns = ['prediction', 'nombre']
#                     fig_bar = px.bar(
#                         counts, x='prediction', y='nombre',
#                         color='prediction',
#                         color_discrete_sequence=[LIGHT_BLUE, ACCENT_BLUE, PRIMARY_MID],
#                         text='nombre',
#                         title="Répartition des prédictions"
#                     )
#                     fig_bar.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         font={'color': TEXT_LIGHT},
#                         title_font={'color': TEXT_LIGHT},
#                         legend_title_text=''
#                     )
#                     st.plotly_chart(fig_bar, use_container_width=True)

#                     csv_bytes = df_result.to_csv(index=False).encode('utf-8')
#                     st.download_button(
#                         "⬇️ Télécharger les résultats (CSV)",
#                         data=csv_bytes,
#                         file_name="predictions.csv",
#                         mime="text/csv",
#                         use_container_width=True
#                     )
#         except Exception as e:
#             st.error(f"Erreur lors de la lecture du fichier : {e}")
#     else:
#         st.info("Importez un fichier CSV pour lancer des prédictions par lot.")

# # ============================================================
# # PIED DE PAGE
# # ============================================================
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown(
#     f"<p style='text-align:center; color:{LIGHT_BLUE}; font-size:0.85rem;'>"
#     "© 2026 — Application de scoring marketing bancaire • Modèle Gradient Boosting"
#     "</p>",
#     unsafe_allow_html=True
# )

#     'contact', 'month', 'day_of_week', 'duration', 'campaign',
#     'pdays', 'previous', 'poutcome'
# ]

# CAT_INDEX = {
#     'job': 0, 'marital': 1, 'education': 2, 'housing': 3,
#     'loan': 4, 'contact': 5, 'month': 6, 'day_of_week': 7, 'poutcome': 8,
# }

# LOAD_ERROR = ""


# @st.cache_resource(show_spinner="Chargement du modèle...")
# def load_resources():
#     encoders = jb.load(BASE_DIR / 'encodes.joblib')
#     cat_goriell = jb.load(BASE_DIR / 'cat_goriell.joblib')
#     scaler = jb.load(BASE_DIR / 'scaler.joblib')
#     gb = jb.load(BASE_DIR / 'gb_model.joblib')
#     clasnames = cat_goriell[9]
#     return encoders, cat_goriell, scaler, gb, clasnames


# try:
#     encoders, cat_goriell, scaler, gb, clasnames = load_resources()
#     RESOURCES_OK = True
# except Exception as e:
#     RESOURCES_OK = False
#     LOAD_ERROR = f"{type(e).__name__}: {e}"


# # ============================================================
# # FONCTIONS LOGIQUE
# # ============================================================

# def build_feature_vector(age, job, marital, education, housing, loan, contact,
#                          month, day_of_week, duration, campaign, pdays,
#                          previous, poutcome):
#     job_enc = encoders[0].transform([job])[0]
#     marital_enc = encoders[1].transform([marital])[0]
#     education_enc = encoders[2].transform([education])[0]
#     housing_enc = encoders[3].transform([housing])[0]
#     loan_enc = encoders[4].transform([loan])[0]
#     contact_enc = encoders[5].transform([contact])[0]
#     month_enc = encoders[6].transform([month])[0]
#     day_of_week_enc = encoders[7].transform([day_of_week])[0]
#     poutcome_enc = encoders[8].transform([poutcome])[0]

#     x_new = np.array([
#         age, job_enc, marital_enc, education_enc, housing_enc, loan_enc,
#         contact_enc, month_enc, day_of_week_enc, duration, campaign, pdays,
#         previous, poutcome_enc
#     ], dtype=float).reshape(1, -1)

#     return scaler.transform(x_new)


# def Pred_func(age, job, marital, education, housing, loan, contact, month,
#               day_of_week, duration, campaign, pdays, previous, poutcome,
#               return_proba=False):
#     x_new = build_feature_vector(
#         age, job, marital, education, housing, loan, contact, month,
#         day_of_week, duration, campaign, pdays, previous, poutcome
#     )
#     y_pred = gb.predict(x_new)[0]
#     label = clasnames[int(y_pred)]
#     if return_proba and hasattr(gb, "predict_proba"):
#         proba = gb.predict_proba(x_new)[0]
#         return label, proba
#     return label, None


# def Pred_func_csv(df):
#     predictions = []
#     confiances = []
#     has_proba = hasattr(gb, "predict_proba")
#     for row in df.itertuples(index=False):
#         try:
#             label, proba = Pred_func(
#                 row.age, row.job, row.marital, row.education, row.housing,
#                 row.loan, row.contact, row.month, row.day_of_week,
#                 row.duration, row.campaign, row.pdays, row.previous,
#                 row.poutcome, return_proba=has_proba
#             )
#             predictions.append(label)
#             confiances.append(
#                 round(float(max(proba)) * 100, 1) if proba is not None else None
#             )
#         except Exception as e:
#             predictions.append(f"Erreur: {e}")
#             confiances.append(None)

#     df_out = df.copy()
#     df_out['prediction'] = predictions
#     if has_proba:
#         df_out['confiance_%'] = confiances
#     return df_out


# # ============================================================
# # EN-TÊTE PRINCIPAL
# # ============================================================

# st.title("🏦 Marketing Bancaire")
# st.subheader("Prédiction des souscriptions")
# st.markdown(
#     "Anticipez, avec un modèle **Gradient Boosting**, "
#     "si un client souscrira à un dépôt à terme et optimisez l'efficacité de vos campagnes."
# )
# st.divider()


# # ============================================================
# # VÉRIFICATION RESSOURCES
# # ============================================================

# if not RESOURCES_OK:
#     st.error(f"Erreur de chargement des fichiers `.joblib` : {LOAD_ERROR}")
#     st.info(
#         "Vérifiez que `encodes.joblib`, `cat_goriell.joblib`, `scaler.joblib` "
#         "et `gb_model.joblib` sont bien à la racine du dépôt (même dossier que "
#         "ce script) et que la version de scikit-learn du `requirements.txt` "
#         "correspond à celle utilisée pour entraîner le modèle."
#     )
#     st.stop()


# # ============================================================
# # BARRE LATÉRALE
# # ============================================================

# with st.sidebar:
#     st.title("🏦 Marketing AI")
#     st.caption("Système prédictif bancaire")
#     st.divider()

#     st.markdown("### ℹ️ À propos")
#     st.write(
#         "Ce tableau de bord utilise un modèle de machine learning basé sur "
#         "**Gradient Boosting**, entraîné sur des données de campagnes marketing "
#         "bancaires afin de prédire la souscription d’un client à un dépôt à terme."
#     )

#     st.markdown("### 🎯 Classes cible")
#     for c in clasnames:
#         st.markdown(f"• **{c}**")

#     st.divider()
#     st.markdown("### 🧩 Variables")
#     st.write(", ".join(COLONNES))


# # ============================================================
# # ONGLETS PRINCIPAUX
# # ============================================================

# tab1, tab2 = st.tabs(["🧑‍💼 Prédiction simple", "📂 Prédiction multiple (CSV)"])


# # ============================================================
# # ONGLET 1 — PRÉDICTION SIMPLE
# # ============================================================

# with tab1:
#     col_form, col_result = st.columns([1.3, 1], gap="large")

#     with col_form:
#         st.subheader("👤 Informations sur le client")

#         c1, c2 = st.columns(2)
#         with c1:
#             age = st.number_input("Âge", min_value=17, max_value=100, value=35)
#             job = st.selectbox("Emploi", cat_goriell[CAT_INDEX['job']])
#             marital = st.selectbox("Statut matrimonial", cat_goriell[CAT_INDEX['marital']])
#             education = st.selectbox("Éducation", cat_goriell[CAT_INDEX['education']])
#             housing = st.selectbox("Prêt immobilier", cat_goriell[CAT_INDEX['housing']])
#             loan = st.selectbox("Prêt personnel", cat_goriell[CAT_INDEX['loan']])
#             contact = st.selectbox("Type de contact", cat_goriell[CAT_INDEX['contact']])

#         with c2:
#             month = st.selectbox("Mois du dernier contact", cat_goriell[CAT_INDEX['month']])
#             day_of_week = st.selectbox("Jour de la semaine", cat_goriell[CAT_INDEX['day_of_week']])
#             duration = st.number_input("Durée du dernier appel (s)", min_value=0, value=180)
#             campaign = st.number_input("Nb de contacts (campagne)", min_value=1, value=2)
#             pdays = st.number_input("Jours depuis dernier contact (999 = jamais)", min_value=0, value=999)
#             previous = st.number_input("Contacts avant cette campagne", min_value=0, value=0)
#             poutcome = st.selectbox("Résultat campagne précédente", cat_goriell[CAT_INDEX['poutcome']])

#         predict_btn = st.button("🔍 Lancer la prédiction", use_container_width=True)

#     with col_result:
#         st.subheader("🎯 Résultat de la prédiction")

#         if predict_btn:
#             try:
#                 label, proba = Pred_func(
#                     age, job, marital, education, housing, loan, contact,
#                     month, day_of_week, duration, campaign, pdays, previous,
#                     poutcome, return_proba=True
#                 )

#                 is_yes = str(label).lower() in ("yes", "oui", "1", "true")
#                 css_class = "result-oui" if is_yes else "result-non"
#                 icon = "✅" if is_yes else "❌"

#                 st.markdown(
#                     f'<div class="result-banner {css_class}">{icon} Souscription prédite : <u>{label}</u></div>',
#                     unsafe_allow_html=True,
#                 )

#                 if proba is not None:
#                     confiance = round(float(max(proba)) * 100, 1)
#                     st.metric("🎯 Confiance du modèle", f"{confiance} %")

#                     fig = go.Figure(
#                         go.Indicator(
#                             mode="gauge+number",
#                             value=confiance,
#                             number={'suffix': " %", 'font': {'color': "#FFFFFF", 'size': 32}},
#                             gauge={
#                                 'axis': {'range': [0, 100], 'tickcolor': "#FFFFFF"},
#                                 'bar': {'color': "#EF4444"},
#                                 'bgcolor': "#3B0505",
#                                 'bordercolor': "#FCA5A5",
#                                 'borderwidth': 2,
#                                 'steps': [
#                                     {'range': [0, 50], 'color': "#450A0A"},
#                                     {'range': [50, 80], 'color': "#7F1D1D"},
#                                     {'range': [80, 100], 'color': "#16A34A"},
#                                 ],
#                             },
#                         )
#                     )
#                     fig.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         font={'color': "#FFFFFF"},
#                         margin=dict(l=20, r=20, t=30, b=20),
#                         height=250,
#                     )
#                     st.plotly_chart(fig, use_container_width=True)

#             except Exception as e:
#                 st.error(f"Erreur lors de la prédiction : {e}")


# # ============================================================
# # ONGLET 2 — PRÉDICTION MULTIPLE (CSV)
# # ============================================================

# with tab2:
#     st.subheader("📂 Importer un fichier CSV pour prédiction par lot")

#     uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

#     if uploaded_file is not None:
#         try:
#             df_input = pd.read_csv(uploaded_file)
#             missing_cols = [c for c in COLONNES if c not in df_input.columns]

#             if missing_cols:
#                 st.error(f"Le fichier CSV manque des colonnes requises : {missing_cols}")
#             else:
#                 st.write("### Aperçu des données importées")
#                 st.dataframe(df_input.head(), use_container_width=True)

#                 if st.button("🚀 Lancer les prédictions sur le CSV"):
#                     with st.spinner("Calcul des prédictions..."):
#                         df_result = Pred_func_csv(df_input)
#                     st.success("Prédictions terminées !")
#                     st.dataframe(df_result, use_container_width=True)

#                     csv_data = df_result.to_csv(index=False).encode('utf-8')
#                     st.download_button(
#                         label="📥 Télécharger les résultats (CSV)",
#                         data=csv_data,
#                         file_name="predictions_marketing.csv",
#                         mime="text/csv",
#                     )
#         except Exception as e:
#             st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
import sys
from pathlib import Path

import streamlit as st

# ============================================================
# CONFIGURATION GÉNÉRALE DE LA PAGE (doit être la 1re commande st.*)
# ============================================================

st.set_page_config(
    page_title="Marketing Bancaire | Prédiction de souscription",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Empêche la traduction automatique par le navigateur (qui casse le DOM React)
st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

# ============================================================
# IMPORTS SÉCURISÉS (message clair si un module manque)
# ============================================================

try:
    import numpy as np
    import pandas as pd
    import joblib as jb
    import plotly.graph_objects as go
except ModuleNotFoundError as e:
    st.error(
        f"Module Python manquant : **{e.name}**.\n\n"
        "Ajoutez un fichier `requirements.txt` à la racine de votre dépôt GitHub "
        "avec : streamlit, numpy, pandas, joblib, scikit-learn, plotly — "
        "puis redémarrez l'application (Manage app > Reboot)."
    )
    st.stop()


# ============================================================
# DESIGN CSS — ROUGE & BLANC
# ============================================================

st.markdown(
    """
<style>

    .stApp {
        background: linear-gradient(135deg, #3B0505 0%, #650B0B 45%, #991B1B 100%) !important;
        color: #FFFFFF !important;
    }

    .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
        color: #FFFFFF !important;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3B0505 0%, #650B0B 50%, #991B1B 100%) !important;
        border-right: 2px solid #EF4444;
    }

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
        background-color: transparent !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stMarkdownContainer"] {
        background-color: transparent !important;
    }

    button[data-baseweb="tab"] {
        background: rgba(69, 10, 10, 0.90) !important;
        color: #FFFFFF !important;
        border-radius: 12px 12px 0 0;
        font-weight: 700;
        padding: 11px 22px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #EF4444, #B91C1C) !important;
        color: #FFFFFF !important;
        border: 1px solid #FCA5A5;
    }

    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #3F0A0A !important;
        border: 2px solid #FCA5A5 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .stSelectbox div[data-baseweb="select"] span {
        color: #3F0A0A !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #EF4444, #B91C1C) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: 1px solid #FCA5A5;
        border-radius: 12px;
        min-height: 48px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #F87171, #DC2626) !important;
    }

    .result-banner {
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 10px;
    }

    .result-oui {
        background: linear-gradient(135deg, #16A34A, #15803D);
        color: #FFFFFF !important;
        border: 2px solid #86EFAC;
    }

    .result-non {
        background: linear-gradient(135deg, #7F1D1D, #450A0A);
        color: #FFFFFF !important;
        border: 2px solid #FCA5A5;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CHARGEMENT DES RESSOURCES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

COLONNES = [
    'age', 'job', 'marital', 'education', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'duration', 'campaign',
    'pdays', 'previous', 'poutcome'
]

CAT_INDEX = {
    'job': 0, 'marital': 1, 'education': 2, 'housing': 3,
    'loan': 4, 'contact': 5, 'month': 6, 'day_of_week': 7, 'poutcome': 8,
}

LOAD_ERROR = ""


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_resources():
    encoders = jb.load(BASE_DIR / 'encodes.joblib')
    cat_goriell = jb.load(BASE_DIR / 'cat_goriell.joblib')
    scaler = jb.load(BASE_DIR / 'scaler.joblib')
    gb = jb.load(BASE_DIR / 'gb_model.joblib')
    clasnames = cat_goriell[9] if len(cat_goriell) > 9 else getattr(gb, "classes_", ["no", "yes"])
    return encoders, cat_goriell, scaler, gb, clasnames


try:
    encoders, cat_goriell, scaler, gb, clasnames = load_resources()
    RESOURCES_OK = True
except Exception as e:
    RESOURCES_OK = False
    LOAD_ERROR = f"{type(e).__name__}: {e}"


# ============================================================
# FONCTIONS LOGIQUE
# ============================================================

def build_feature_vector(age, job, marital, education, housing, loan, contact,
                         month, day_of_week, duration, campaign, pdays,
                         previous, poutcome):
    job_enc = encoders[0].transform([job])[0]
    marital_enc = encoders[1].transform([marital])[0]
    education_enc = encoders[2].transform([education])[0]
    housing_enc = encoders[3].transform([housing])[0]
    loan_enc = encoders[4].transform([loan])[0]
    contact_enc = encoders[5].transform([contact])[0]
    month_enc = encoders[6].transform([month])[0]
    day_of_week_enc = encoders[7].transform([day_of_week])[0]
    poutcome_enc = encoders[8].transform([poutcome])[0]

    x_new = np.array([
        age, job_enc, marital_enc, education_enc, housing_enc, loan_enc,
        contact_enc, month_enc, day_of_week_enc, duration, campaign, pdays,
        previous, poutcome_enc
    ], dtype=float).reshape(1, -1)

    return scaler.transform(x_new)


def Pred_func(age, job, marital, education, housing, loan, contact, month,
              day_of_week, duration, campaign, pdays, previous, poutcome,
              return_proba=False):
    x_new = build_feature_vector(
        age, job, marital, education, housing, loan, contact, month,
        day_of_week, duration, campaign, pdays, previous, poutcome
    )
    y_pred = gb.predict(x_new)[0]
    label = clasnames[int(y_pred)]
    if return_proba and hasattr(gb, "predict_proba"):
        proba = gb.predict_proba(x_new)[0]
        return label, proba
    return label, None


def Pred_func_csv(df):
    predictions = []
    confiances = []
    has_proba = hasattr(gb, "predict_proba")
    for row in df.itertuples(index=False):
        try:
            label, proba = Pred_func(
                row.age, row.job, row.marital, row.education, row.housing,
                row.loan, row.contact, row.month, row.day_of_week,
                row.duration, row.campaign, row.pdays, row.previous,
                row.poutcome, return_proba=has_proba
            )
            predictions.append(label)
            confiances.append(
                round(float(max(proba)) * 100, 1) if proba is not None else None
            )
        except Exception as e:
            predictions.append(f"Erreur: {e}")
            confiances.append(None)

    df_out = df.copy()
    df_out['prediction'] = predictions
    if has_proba:
        df_out['confiance_%'] = confiances
    return df_out


# ============================================================
# EN-TÊTE PRINCIPAL
# ============================================================

st.title("🏦 Marketing Bancaire")
st.subheader("Prédiction des souscriptions")
st.markdown(
    "Anticipez, avec un modèle **Gradient Boosting**, "
    "si un client souscrira à un dépôt à terme et optimisez l'efficacité de vos campagnes."
)
st.divider()


# ============================================================
# VÉRIFICATION RESSOURCES
# ============================================================

if not RESOURCES_OK:
    st.error(f"Erreur de chargement des fichiers `.joblib` : {LOAD_ERROR}")
    st.info(
        "Vérifiez que `encodes.joblib`, `cat_goriell.joblib`, `scaler.joblib` "
        "et `gb_model.joblib` sont bien à la racine du dépôt (même dossier que "
        "ce script) et que la version de scikit-learn du `requirements.txt` "
        "correspond à celle utilisée pour entraîner le modèle."
    )
    st.stop()


# ============================================================
# BARRE LATÉRALE
# ============================================================

with st.sidebar:
    st.title("🏦 Marketing AI")
    st.caption("Système prédictif bancaire")
    st.divider()

    st.markdown("### ℹ️ À propos")
    st.write(
        "Ce tableau de bord utilise un modèle de machine learning basé sur "
        "**Gradient Boosting**, entraîné sur des données de campagnes marketing "
        "bancaires afin de prédire la souscription d’un client à un dépôt à terme."
    )

    st.markdown("### 🎯 Classes cible")
    for c in clasnames:
        st.markdown(f"• **{c}**")

    st.divider()
    st.markdown("### 🧩 Variables")
    st.write(", ".join(COLONNES))


# ============================================================
# ONGLETS PRINCIPAUX
# ============================================================

tab1, tab2 = st.tabs(["🧑‍💼 Prédiction simple", "📂 Prédiction multiple (CSV)"])


# ============================================================
# ONGLET 1 — PRÉDICTION SIMPLE
# ============================================================

with tab1:
    col_form, col_result = st.columns([1.3, 1], gap="large")

    with col_form:
        st.subheader("👤 Informations sur le client")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Âge", min_value=17, max_value=100, value=35)
            job = st.selectbox("Emploi", cat_goriell[CAT_INDEX['job']])
            marital = st.selectbox("Statut matrimonial", cat_goriell[CAT_INDEX['marital']])
            education = st.selectbox("Éducation", cat_goriell[CAT_INDEX['education']])
            housing = st.selectbox("Prêt immobilier", cat_goriell[CAT_INDEX['housing']])
            loan = st.selectbox("Prêt personnel", cat_goriell[CAT_INDEX['loan']])
            contact = st.selectbox("Type de contact", cat_goriell[CAT_INDEX['contact']])

        with c2:
            month = st.selectbox("Mois du dernier contact", cat_goriell[CAT_INDEX['month']])
            day_of_week = st.selectbox("Jour de la semaine", cat_goriell[CAT_INDEX['day_of_week']])
            duration = st.number_input("Durée du dernier appel (s)", min_value=0, value=180)
            campaign = st.number_input("Nb de contacts (campagne)", min_value=1, value=2)
            pdays = st.number_input("Jours depuis dernier contact (999 = jamais)", min_value=0, value=999)
            previous = st.number_input("Contacts avant cette campagne", min_value=0, value=0)
            poutcome = st.selectbox("Résultat campagne précédente", cat_goriell[CAT_INDEX['poutcome']])

        predict_btn = st.button("🔍 Lancer la prédiction", use_container_width=True)

    with col_result:
        st.subheader("🎯 Résultat de la prédiction")

        if predict_btn:
            try:
                label, proba = Pred_func(
                    age, job, marital, education, housing, loan, contact,
                    month, day_of_week, duration, campaign, pdays, previous,
                    poutcome, return_proba=True
                )

                is_yes = str(label).lower() in ("yes", "oui", "1", "true")
                css_class = "result-oui" if is_yes else "result-non"
                icon = "✅" if is_yes else "❌"

                # Utilisation d'un conteneur natif Streamlit pour éviter la manipulation directe du DOM
                result_container = st.container()
                with result_container:
                    st.markdown(
                        f'<div class="result-banner {css_class}">{icon} Souscription prédite : <u>{label}</u></div>',
                        unsafe_allow_html=True,
                    )

                if proba is not None:
                    confiance = round(float(max(proba)) * 100, 1)
                    st.metric("🎯 Confiance du modèle", f"{confiance} %")

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=confiance,
                            number={'suffix': " %", 'font': {'color': "#FFFFFF", 'size': 32}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': "#FFFFFF"},
                                'bar': {'color': "#EF4444"},
                                'bgcolor': "#3B0505",
                                'bordercolor': "#FCA5A5",
                                'borderwidth': 2,
                                'steps': [
                                    {'range': [0, 50], 'color': "#450A0A"},
                                    {'range': [50, 80], 'color': "#7F1D1D"},
                                    {'range': [80, 100], 'color': "#16A34A"},
                                ],
                            },
                        )
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#FFFFFF"},
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=250,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")


# ============================================================
# ONGLET 2 — PRÉDICTION MULTIPLE (CSV)
# ============================================================

with tab2:
    st.subheader("📂 Importer un fichier CSV pour prédiction par lot")

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            missing_cols = [c for c in COLONNES if c not in df_input.columns]

            if missing_cols:
                st.error(f"Le fichier CSV manque des colonnes requises : {missing_cols}")
            else:
                st.write("### Aperçu des données importées")
                st.dataframe(df_input.head(), use_container_width=True)

                if st.button("🚀 Lancer les prédictions sur le CSV"):
                    with st.spinner("Calcul des prédictions..."):
                        df_result = Pred_func_csv(df_input)
                    st.success("Prédictions terminées !")
                    st.dataframe(df_result, use_container_width=True)

                    csv_data = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les résultats (CSV)",
                        data=csv_data,
                        file_name="predictions_marketing.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
