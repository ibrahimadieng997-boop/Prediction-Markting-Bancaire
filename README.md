<div align="center">

# 🏦 Marketing Bancaire — Prédiction de souscription

### Quel client va souscrire à un dépôt à terme ? Le Machine Learning répond.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Gradient Boosting](https://img.shields.io/badge/Gradient%20Boosting-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)

</div>

---

## 🎯 Le projet en 30 secondes

Les banques contactent des milliers de clients pour leur proposer un **dépôt à terme**. Appeler tout le monde coûte cher et agace les clients peu intéressés.

**Ce projet prédit, avant l'appel, si un client va souscrire**, à partir de 14 informations : profil, historique de contact et résultat des campagnes précédentes. Résultat : des campagnes mieux ciblées, moins de temps perdu, un meilleur taux de conversion.

L'ensemble est livré sous forme d'**application web interactive**, utilisable par une équipe marketing sans aucune compétence technique.

## ✨ Fonctionnalités

| | Fonctionnalité | Description |
|---|---|---|
| 🧑‍💼 | **Prédiction simple** | Formulaire client → verdict *oui / non* en un clic |
| 🎯 | **Score de confiance** | Jauge interactive du niveau de certitude du modèle |
| 📂 | **Prédiction par lot** | Import d'un CSV, prédiction de toute la liste clients |
| 📊 | **Visualisation** | Répartition des prédictions (graphique Plotly) |
| 📥 | **Export** | Téléchargement des résultats au format CSV |
| 🎨 | **Interface dédiée** | Thème vert et or, barre latérale d'aide, messages d'erreur clairs |

## 🧠 Démarche Data Science

| Étape | Ce qui a été réalisé |
|---|---|
| **Données** | 11 640 clients, 14 variables (numériques et catégorielles) |
| **Pré-traitement** | Encodage de 9 variables catégorielles, normalisation (scaler) |
| **Modélisation** | Classification binaire avec **Gradient Boosting** |
| **Industrialisation** | Modèle, encodeurs et scaler sérialisés (Joblib) |
| **Déploiement** | Application Streamlit avec prédiction unitaire et par lot |

### Variables utilisées

| Catégorie | Variables |
|---|---|
| 👤 **Profil client** | `age`, `job`, `marital`, `education` |
| 💳 **Situation bancaire** | `housing` (prêt immobilier), `loan` (prêt personnel) |
| 📞 **Dernier contact** | `contact`, `month`, `day_of_week`, `duration` |
| 📈 **Historique campagnes** | `campaign`, `pdays`, `previous`, `poutcome` |

## 🛠️ Compétences démontrées

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Gradient Boosting` · `Classification binaire` · `Feature engineering (encodage, normalisation)` · `Plotly` · `Streamlit` · `Joblib` · `Gestion d'erreurs et validation des données` · `HTML/CSS`

## 🚀 Lancer le projet
```
Lancer l'application
```

[Lien vers l'application: https://prediction-markting-bancaire-txovzczirswkhaendycm3w.streamlit.app/].

> 💡 Pour tester la prédiction par lot, utilisez directement le fichier d'exemple `df_copy.csv` fourni dans le dépôt.

## 📁 Format du CSV (prédiction par lot)

Colonnes requises, dans n'importe quel ordre :

```
age, job, marital, education, housing, loan, contact,
month, day_of_week, duration, campaign, pdays, previous, poutcome
```

Exemple :

```csv
age,job,marital,education,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome
28,management,single,university.degree,yes,no,cellular,jun,thu,339,3,6,2,success
55,retired,married,basic.4y,yes,no,cellular,aug,fri,137,1,3,1,success
```

Le fichier de sortie ajoute les colonnes **`prediction`** et **`confiance_%`**.

## 🗂️ Structure du projet

```
├── app.py               # Application Streamlit
├── gb_model.joblib      # Modèle Gradient Boosting entraîné
├── encodes.joblib       # Encodeurs des variables catégorielles
├── cat_goriell.joblib   # Modalités des variables et classes cibles
├── scaler.joblib        # Normalisation des données
├── df_copy.csv          # Données d'exemple pour tester le CSV
└── requirements.txt
```

## 🔮 Pistes d'amélioration

- Afficher l'importance des variables (explicabilité du modèle)
- Comparer plusieurs algorithmes (Random Forest, XGBoost, régression logistique)
- Ajouter les métriques d'évaluation (précision, rappel, AUC) dans l'application
- Déployer en ligne sur Streamlit Community Cloud

## 👤 Auteur

**Ibrahima Dieng**
Junior en Data Science et Intelligence Artificielle

📍 Thiès, Médina Fall, Sénégal
💻 GitHub : [ibrahimadieng997boop](https://github.com/ibrahimadieng997boop)
✉️ [ibrahimadieng997@gmail.com](mailto:ibrahimadieng997@gmail.com)

---

<div align="center">

⭐ *Ce projet vous plaît ? Laissez une étoile, et n'hésitez pas à me contacter !*

</div>
