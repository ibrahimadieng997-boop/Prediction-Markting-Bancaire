from pathlib import Path

import joblib as jb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURATION GÉNÉRALE DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Marketing Bancaire | Prédiction de souscription",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# THÈME — VERT FONCÉ / JAUNE FONCÉ (contrastes élevés)
# ============================================================
BG_DARK = "#0A2A1A"      # fond principal : vert très foncé
BG_MID = "#12402A"       # vert foncé intermédiaire (cartes, sidebar)
GREEN = "#3F8F4F"        # vert moyen (bordures, accents)
GREEN_LIGHT = "#7CC48A"  # vert clair (valeurs, détails)
YELLOW = "#E6B800"       # jaune foncé / doré (titres, boutons)
YELLOW_LIGHT = "#F4D35E" # jaune plus clair (survols, jauge)
CREAM = "#FFF8DC"        # texte clair sur fond foncé
INPUT_BG = "#FFF8DC"     # fond des champs de saisie (clair)
INPUT_TEXT = "#0A2A1A"   # texte des champs de saisie (foncé)

st.markdown(f"""
<style>
    /* ---------- Fond général ---------- */
    .stApp {{
        background: linear-gradient(180deg, {BG_DARK} 0%, {BG_MID} 100%);
        color: {CREAM};
    }}
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* ---------- Textes ---------- */
    .stApp p, .stApp li, .stApp label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p {{
        color: {CREAM};
    }}
    [data-testid="stCaptionContainer"] {{
        color: {GREEN_LIGHT};
    }}
    h1, h2, h3, h4 {{
        color: {YELLOW} !important;
        font-weight: 800 !important;
    }}

    /* ---------- Barre latérale ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BG_DARK} 0%, {BG_MID} 100%);
        border-right: 2px solid {YELLOW};
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span {{
        color: {CREAM} !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {YELLOW} !important;
    }}

    /* ---------- Cartes (conteneurs avec bordure) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: rgba(230, 184, 0, 0.06);
        border: 1px solid {GREEN};
        border-radius: 16px;
    }}

    /* ---------- Onglets ---------- */
    button[data-baseweb="tab"] {{
        background-color: {BG_MID};
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        margin-right: 4px;
    }}
    button[data-baseweb="tab"] p {{
        color: {CREAM} !important;
        font-weight: 600;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {YELLOW};
    }}
    button[data-baseweb="tab"][aria-selected="true"] p {{
        color: {BG_DARK} !important;
        font-weight: 800;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {YELLOW_LIGHT} !important;
    }}

    /* ---------- Boutons ---------- */
    .stButton > button,
    .stDownloadButton > button {{
        width: 100%;
        background: linear-gradient(90deg, {YELLOW}, {YELLOW_LIGHT});
        color: {BG_DARK} !important;
        font-weight: 800;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        transition: 0.2s;
    }}
    .stButton > button p,
    .stDownloadButton > button p {{
        color: {BG_DARK} !important;
        font-weight: 800;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        transform: scale(1.02);
        box-shadow: 0 0 12px {YELLOW}AA;
    }}

    /* ---------- Champs numériques ---------- */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {{
        background-color: {INPUT_BG} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="input"] input {{
        color: {INPUT_TEXT} !important;
        -webkit-text-fill-color: {INPUT_TEXT} !important;
        font-weight: 600;
    }}
    div[data-testid="stNumberInput"] button {{
        background-color: #E9DFA6 !important;
        color: {INPUT_TEXT} !important;
    }}
    div[data-testid="stNumberInput"] button svg {{
        fill: {INPUT_TEXT} !important;
    }}

    /* ---------- Listes déroulantes ---------- */
    div[data-baseweb="select"] > div {{
        background-color: {INPUT_BG} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] * {{
        color: {INPUT_TEXT} !important;
        font-weight: 600;
    }}
    div[data-baseweb="select"] svg {{
        fill: {INPUT_TEXT} !important;
    }}
    div[data-baseweb="popover"] ul,
    ul[role="listbox"] {{
        background-color: {INPUT_BG} !important;
    }}
    ul[role="listbox"] li,
    ul[role="listbox"] li * {{
        color: {INPUT_TEXT} !important;
    }}
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] {{
        background-color: {YELLOW_LIGHT} !important;
    }}

    /* ---------- Import de fichier ---------- */
    section[data-testid="stFileUploaderDropzone"] {{
        background-color: {BG_MID};
        border: 2px dashed {YELLOW};
        border-radius: 12px;
    }}
    section[data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] small {{
        color: {CREAM} !important;
    }}
    section[data-testid="stFileUploaderDropzone"] button {{
        background: {YELLOW};
        border: none;
    }}
    section[data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploaderDropzone"] button * {{
        color: {BG_DARK} !important;
        font-weight: 700;
    }}

    /* ---------- Métriques ---------- */
    div[data-testid="stMetric"] {{
        background-color: rgba(230, 184, 0, 0.10);
        border: 1px solid {YELLOW};
        border-radius: 14px;
        padding: 14px;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {CREAM} !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {YELLOW_LIGHT} !important;
    }}

    /* ---------- Messages (info / succès / erreur) ---------- */
    div[data-testid="stAlert"] {{
        background-color: rgba(230, 184, 0, 0.12);
        border: 1px solid {YELLOW};
        border-radius: 12px;
    }}
    div[data-testid="stAlert"] * {{
        color: {CREAM} !important;
    }}

    /* ---------- Bannière résultat ---------- */
    .result-banner {{
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
        margin: 10px 0 14px 0;
    }}
    .result-oui {{
        background: linear-gradient(90deg, {YELLOW}, {YELLOW_LIGHT});
        color: {BG_DARK};
    }}
    .result-non {{
        background: linear-gradient(90deg, {BG_MID}, {GREEN});
        color: {CREAM};
        border: 2px solid {YELLOW};
    }}

    /* ---------- Séparateurs ---------- */
    hr {{
        border-color: {GREEN};
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTES DU MODÈLE
# ============================================================
COLONNES = [
    'age', 'job', 'marital', 'education', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'duration', 'campaign',
    'pdays', 'previous', 'poutcome',
]

CAT_INDEX = {
    'job': 0, 'marital': 1, 'education': 2, 'housing': 3, 'loan': 4,
    'contact': 5, 'month': 6, 'day_of_week': 7, 'poutcome': 8,
}

LOAD_ERROR = ""


# ============================================================
# CHARGEMENT DES RESSOURCES (mise en cache)
# ============================================================
@st.cache_resource(show_spinner="Chargement du modèle et des encodeurs...")
def load_resources():
    encoders = jb.load(BASE_DIR / 'encodes.joblib')
    cat_goriell = jb.load(BASE_DIR / 'cat_goriell.joblib')
    scaler = jb.load(BASE_DIR / 'scaler.joblib')
    gb = jb.load(BASE_DIR / 'gb_model.joblib')
    clasnames = cat_goriell[9]
    return encoders, cat_goriell, scaler, gb, clasnames


try:
    encoders, cat_goriell, scaler, gb, clasnames = load_resources()
    RESOURCES_OK = True
except Exception as e:
    RESOURCES_OK = False
    LOAD_ERROR = f"{type(e).__name__}: {e}"


# ============================================================
# FONCTIONS DE PRÉDICTION
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
        previous, poutcome_enc,
    ], dtype=float).reshape(1, -1)

    return scaler.transform(x_new)


def Pred_func(age, job, marital, education, housing, loan, contact, month,
              day_of_week, duration, campaign, pdays, previous, poutcome,
              return_proba=False):
    x_new = build_feature_vector(
        age, job, marital, education, housing, loan, contact, month,
        day_of_week, duration, campaign, pdays, previous, poutcome,
    )
    y_pred = gb.predict(x_new)[0]

    # La prédiction peut être un indice (0/1) ou déjà un libellé texte
    try:
        label = clasnames[int(y_pred)]
    except (ValueError, TypeError):
        label = str(y_pred)

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
                row.poutcome, return_proba=has_proba,
            )
            predictions.append(str(label))
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
st.markdown(f"""
<div style="text-align:center; padding: 10px 0 10px 0;">
    <h1>🏦 Marketing Bancaire — Prédiction des souscriptions</h1>
    <p style="color:{CREAM}; font-size:1.05rem;">
        Anticipez, avec un modèle <b style="color:{YELLOW_LIGHT};">Gradient Boosting</b>,
        si un client souscrira à un dépôt à terme et optimisez l'efficacité de vos campagnes.
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ============================================================
# VÉRIFICATION DES RESSOURCES
# ============================================================
if not RESOURCES_OK:
    st.error(f"Erreur de chargement des fichiers `.joblib` : {LOAD_ERROR}")
    st.info(
        "Vérifiez que `encodes.joblib`, `cat_goriell.joblib`, `scaler.joblib` "
        "et `gb_model.joblib` sont bien dans le même dossier que ce script "
        "et que la version de scikit-learn du `requirements.txt` correspond "
        "à celle utilisée pour entraîner le modèle."
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
        "bancaires afin de prédire la souscription d'un client à un dépôt à terme."
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

# ------------------------------------------------------------
# ONGLET 1 — PRÉDICTION SIMPLE
# ------------------------------------------------------------
with tab1:
    col_form, col_result = st.columns([1.3, 1], gap="large")

    with col_form:
        with st.container(border=True):
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

            predict_btn = st.button("🔍 Lancer la prédiction")

    with col_result:
        with st.container(border=True):
            st.subheader("🎯 Résultat de la prédiction")

            if predict_btn:
                try:
                    label, proba = Pred_func(
                        age, job, marital, education, housing, loan, contact,
                        month, day_of_week, duration, campaign, pdays, previous,
                        poutcome, return_proba=True,
                    )

                    is_yes = str(label).strip().lower() in ("yes", "oui", "1", "true")
                    css_class = "result-oui" if is_yes else "result-non"
                    icon = "✅" if is_yes else "❌"

                    st.markdown(
                        f'<div class="result-banner {css_class}">'
                        f'{icon} Souscription prédite : <u>{label}</u></div>',
                        unsafe_allow_html=True,
                    )

                    if proba is not None:
                        confiance = round(float(max(proba)) * 100, 1)
                        st.metric("🎯 Confiance du modèle", f"{confiance} %")

                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=confiance,
                                number={'suffix': " %", 'font': {'color': CREAM, 'size': 34}},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickcolor': CREAM,
                                             'tickfont': {'color': CREAM}},
                                    'bar': {'color': YELLOW},
                                    'bgcolor': BG_DARK,
                                    'bordercolor': YELLOW,
                                    'borderwidth': 2,
                                    'steps': [
                                        {'range': [0, 50], 'color': "#1B4D30"},
                                        {'range': [50, 80], 'color': "#2A6B3F"},
                                        {'range': [80, 100], 'color': "#3F8F4F"},
                                    ],
                                },
                            )
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': CREAM},
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=260,
                        )
                        st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")
            else:
                st.info("Renseignez les informations du client puis cliquez sur "
                        "**Lancer la prédiction**.")

# ------------------------------------------------------------
# ONGLET 2 — PRÉDICTION MULTIPLE (CSV)
# ------------------------------------------------------------
with tab2:
    st.subheader("📂 Importer un fichier CSV pour prédiction par lot")
    st.write(
        "Le fichier doit contenir les colonnes suivantes (dans n'importe quel ordre) : "
        f"`{', '.join(COLONNES)}`"
    )

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

    if uploaded_file is None:
        # Nouveau départ : on efface d'anciens résultats
        st.session_state.pop("df_result", None)
        st.session_state.pop("result_file", None)
        st.info("Importez un fichier CSV pour lancer des prédictions par lot.")
    else:
        df_input = None
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier CSV : {e}")

        if df_input is not None:
            missing_cols = [c for c in COLONNES if c not in df_input.columns]

            if missing_cols:
                st.error(f"Le fichier CSV manque des colonnes requises : {missing_cols}")
            else:
                st.success(f"Fichier chargé — {len(df_input)} lignes détectées.")
                st.markdown("### Aperçu des données importées")
                st.dataframe(df_input.head())

                if st.button("🚀 Lancer les prédictions sur le CSV"):
                    with st.spinner("Calcul des prédictions..."):
                        st.session_state["df_result"] = Pred_func_csv(df_input)
                        st.session_state["result_file"] = uploaded_file.name

                # Les résultats restent affichés même après un clic sur « Télécharger »
                if (
                    "df_result" in st.session_state
                    and st.session_state.get("result_file") == uploaded_file.name
                ):
                    df_result = st.session_state["df_result"]

                    st.success("Prédictions terminées !")
                    st.markdown("### 📊 Résultats")
                    st.dataframe(df_result)

                    counts = df_result['prediction'].value_counts().reset_index()
                    counts.columns = ['prediction', 'nombre']
                    fig_bar = px.bar(
                        counts, x='prediction', y='nombre',
                        color='prediction',
                        color_discrete_sequence=[YELLOW, GREEN_LIGHT, GREEN, YELLOW_LIGHT],
                        text='nombre',
                        title="Répartition des prédictions",
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': CREAM},
                        title_font={'color': YELLOW},
                        legend_title_text='',
                        xaxis={'gridcolor': "#1B4D30", 'color': CREAM},
                        yaxis={'gridcolor': "#1B4D30", 'color': CREAM},
                    )
                    st.plotly_chart(fig_bar)

                    csv_data = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les résultats (CSV)",
                        data=csv_data,
                        file_name="predictions_marketing.csv",
                        mime="text/csv",
                    )

# ============================================================
# PIED DE PAGE
# ============================================================
st.divider()
st.markdown(
    f"<p style='text-align:center; color:{GREEN_LIGHT}; font-size:0.85rem;'>"
    "© 2026 — Application de scoring marketing bancaire • Modèle Gradient Boosting"
    "</p>",
    unsafe_allow_html=True,
)
