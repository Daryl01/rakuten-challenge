"""
Application Streamlit - Projet Rakuten
Classification multimodale de produits e-commerce

Auteurs : Khoty WOLIE | [Prénom NOM 2] | [Prénom NOM 3]
Formation Ingénieur IA - Liora | Juillet 2026

Utilisation :
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import joblib
import os
import re
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Rakuten - Classification multimodale",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Chemins du projet ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# Chemins absolus du projet (à adapter si déplacé)
PROJECT_ROOT = ROOT.parent  # remonte d'un niveau si app.py est dans streamlit_app/

PATHS = {
    "svc_model"    : PROJECT_ROOT / "models" / "baselines" / "tfidf_linearsvc.pkl",
    "figures"      : PROJECT_ROOT / "reports" / "figures",
    "data_proc"    : PROJECT_ROOT / "data" / "processed" / "df_train_processed.csv",
    "results_csv"  : PROJECT_ROOT / "models" / "artifacts" / "resultats_finaux_03.csv",
    "submission"   : PROJECT_ROOT / "data" / "submissions" / "submission_tfidf_svc.csv",
    "image_train"  : PROJECT_ROOT / "data" / "raw" / "image_train",
}

# ── Mapping des codes produits ────────────────────────────────────────────────
LABEL_MAP = {
    10:   "Livres grand public",
    40:   "Jeux vidéo - consoles neufs",
    50:   "Accessoires gaming",
    60:   "Consoles",
    1140: "Puériculture - accessoires",
    1160: "Cartes de jeu",
    1180: "Jeux vidéo - rétro",
    1280: "Jouets et peluches",
    1281: "Jeux de société",
    1300: "Jeux miniatures",
    1301: "Chaussures et vêtements",
    1302: "Jeux de plein air",
    1320: "Jeux d'extérieur",
    1560: "Mobilier",
    1920: "Literie et linge de maison",
    1940: "Alimentation et boissons",
    2060: "Décoration intérieure",
    2220: "Animaux domestiques",
    2280: "Presse et magazines",
    2403: "Livres - lots et collections",
    2462: "Jeux vidéo - accessoires",
    2522: "Papeterie et fournitures",
    2582: "Mobilier de bureau",
    2583: "Piscines et spas",
    2585: "Bricolage et jardinage",
    2705: "Livres techniques et manuels",
    2905: "Livres rares et anciens",
}

# ── Styles CSS personnalisés ──────────────────────────────────────────────────
st.markdown("""
<style>
/* Barre latérale */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2a4a 0%, #2E75B6 100%);
}
[data-testid="stSidebar"] * { color: white !important; }

/* Titres principaux */
h1 { color: #2E75B6 !important; border-bottom: 3px solid #2E75B6; padding-bottom: 8px; }
h2 { color: #1a2a4a !important; }
h3 { color: #2E75B6 !important; }

/* Métriques */
[data-testid="metric-container"] {
    background-color: #f0f6ff;
    border: 1px solid #2E75B6;
    border-radius: 8px;
    padding: 12px;
}

/* Encadrés résultat */
.result-box {
    background: linear-gradient(135deg, #2E75B6 0%, #1a2a4a 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0;
}
.result-box h2 { color: white !important; font-size: 28px; margin: 0; }
.result-box p  { color: #cce0ff; margin: 5px 0 0 0; font-size: 14px; }

/* Badge de confiance */
.confidence-high { color: #00a651; font-weight: bold; }
.confidence-med  { color: #f5a623; font-weight: bold; }
.confidence-low  { color: #d0021b; font-weight: bold; }

/* Tableaux */
.stDataFrame { border: 1px solid #2E75B6 !important; }

/* Info box */
.info-rakuten {
    background-color: #e8f0fe;
    border-left: 5px solid #2E75B6;
    padding: 15px 20px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
}

/* Tabs - labels plus grands et gras */
[data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #e0e8f5;
}
[data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    color: #4a5568 !important;
    border-radius: 6px 6px 0 0 !important;
    letter-spacing: 0.01em;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #2E75B6 !important;
    border-bottom: 3px solid #2E75B6 !important;
    background-color: #f0f6ff !important;
}
[data-baseweb="tab"]:hover {
    color: #2E75B6 !important;
    background-color: #f5f9ff !important;
}

/* Sidebar - ameliorations visuelles */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1e38 0%, #1a2e52 40%, #2E75B6 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] .sidebar-section-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.5) !important;
    margin-bottom: 4px;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
    margin: 10px 0 !important;
}
.sidebar-metric-val {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff !important;
    line-height: 1.2;
}
.sidebar-metric-delta {
    font-size: 12px;
    color: #7ec8a0 !important;
    font-weight: 500;
}
.sidebar-label {
    font-size: 11px;
    color: rgba(255,255,255,0.55) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Chargement du modèle (avec cache) ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_svc_model():
    """Charge le pipeline TF-IDF + LinearSVC sauvegardé avec joblib.
    Retourne (pipeline, None) si OK, (None, message_erreur) sinon.
    Le fichier est produit par joblib.dump() dans le notebook 02 (cellule 2.7).
    """
    model_path = PATHS["svc_model"]
    if not model_path.exists():
        return None, f"Fichier introuvable : {model_path}"
    try:
        return joblib.load(model_path), None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def load_dataset():
    """Charge le dataset préprocessé."""
    path = PATHS["data_proc"]
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    # Assurer la présence de la colonne text_combined
    if "text_combined" not in df.columns:
        df["designation_clean"] = df["designation_clean"].fillna("")
        df["description_clean"] = df["description_clean"].fillna("")
        df["text_combined"] = df["designation_clean"] + " " + df["description_clean"]
    return df

@st.cache_data(show_spinner=False)
def load_results():
    """Charge le tableau des résultats finaux."""
    path = PATHS["results_csv"]
    if not path.exists():
        # Tableau de secours codé en dur
        return pd.DataFrame({
            "Modele": [
                "TF-IDF + LinearSVC",
                "Benchmark Rakuten - Texte (CNN)",
                "SBERT + MLP",
                "Fusion SBERT + ResNet50",
                "ResNet50 Phase 2 (fine-tuning)",
                "SBERT + LogisticRegression",
                "ResNet50 Phase 1",
                "Benchmark Rakuten - Image (ResNet50)",
            ],
            "Modalite": ["Texte","Texte","Texte","Texte+Image","Image","Texte","Image","Image"],
            "Weighted_F1": [0.8403, 0.8113, 0.7542, 0.7105, 0.6836, 0.6829, 0.6174, 0.5534],
            "Type": ["Notre modèle","Benchmark","Notre modèle","Notre modèle",
                     "Notre modèle","Notre modèle","Notre modèle","Benchmark"],
        })
    return pd.read_csv(path)

def fig_path(name):
    return PATHS["figures"] / name

def load_figure(filename):
    """Charge une figure PNG depuis reports/figures/."""
    p = fig_path(filename)
    if p.exists():
        return str(p)
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # En-tête projet
    st.markdown("""
    <div style="padding: 8px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom: 16px;">
        <div style="font-size:20px; font-weight:700; color:#ffffff; letter-spacing:0.02em;">Rakuten France</div>
        <div style="font-size:12px; color:rgba(255,255,255,0.6); margin-top:3px; font-weight:400;">
            Classification multimodale de produits
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bloc projet / formation
    st.markdown("""
    <div style="margin-bottom:14px;">
        <div class="sidebar-label">Projet</div>
        <div style="color:#ffffff; font-size:13px; margin-top:3px;">
            Formation Ingénieur IA<br>
            <span style="color:rgba(255,255,255,0.7);">Liora (ex DataScientest)</span>
        </div>
        <div style="color:rgba(255,255,255,0.55); font-size:12px; margin-top:4px;">
            Soutenance : 16 octobre 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:10px 0;'>", unsafe_allow_html=True)

    # Auteurs
    st.markdown("""
    <div style="margin-bottom:14px;">
        <div class="sidebar-label">Auteurs</div>
        <div style="color:#ffffff; font-size:13px; margin-top:5px; line-height:1.9;">
            Khoty WOLIE<br>
            [Prénom NOM 2]<br>
            [Prénom NOM 3]
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:10px 0;'>", unsafe_allow_html=True)

    # Données
    st.markdown("""
    <div style="margin-bottom:14px;">
        <div class="sidebar-label">Données</div>
        <div style="display:flex; justify-content:space-between; margin-top:6px;">
            <div style="text-align:center;">
                <div class="sidebar-metric-val">84 916</div>
                <div style="font-size:10px; color:rgba(255,255,255,0.5);">produits</div>
            </div>
            <div style="text-align:center;">
                <div class="sidebar-metric-val">27</div>
                <div style="font-size:10px; color:rgba(255,255,255,0.5);">classes</div>
            </div>
            <div style="text-align:center;">
                <div class="sidebar-metric-val">2</div>
                <div style="font-size:10px; color:rgba(255,255,255,0.5);">modalités</div>
            </div>
        </div>
        <div style="font-size:11px; color:rgba(255,255,255,0.5); margin-top:6px;">
            Texte + Image &nbsp;|&nbsp; Weighted F1-score
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:10px 0;'>", unsafe_allow_html=True)

    # Meilleur modèle
    st.markdown("""
    <div style="margin-bottom:14px;">
        <div class="sidebar-label">Meilleur modèle</div>
        <div style="margin-top:6px;">
            <div style="color:#ffffff; font-size:13px; font-weight:600;">TF-IDF + LinearSVC</div>
            <div style="margin-top:6px; display:flex; align-items:baseline; gap:6px;">
                <span class="sidebar-metric-val" style="font-size:26px;">0.8403</span>
                <span class="sidebar-metric-delta">+2.9 pts vs benchmark</span>
            </div>
            <div style="margin-top:4px;">
                <span style="background:rgba(46,117,182,0.4); color:#cce0ff; font-size:11px;
                             padding:2px 8px; border-radius:12px; border:1px solid rgba(255,255,255,0.2);">
                    F1 pondéré
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:10px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px; color:rgba(255,255,255,0.4); text-align:center; padding-top:4px;">
        Liora (ex DataScientest) &nbsp;·&nbsp; Octobre 2026
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  ONGLETS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Contexte",
    "Exploration",
    "Modélisation",
    "Classification en direct",
    "Interprétabilité",
])


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 1 - CONTEXTE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Classification multimodale de produits e-commerce Rakuten")
    st.markdown("---")

    col_logo, col_intro = st.columns([1, 2])

    with col_logo:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#2E75B6,#1a2a4a);
                    color:white;padding:30px;border-radius:12px;text-align:center;">
            <h1 style="color:white!important;font-size:48px;border:none;">🛍️</h1>
            <h3 style="color:white!important;">Rakuten France</h3>
            <p style="color:#cce0ff;">Multimodal Product Data Classification</p>
            <br>
            <p style="color:#aad4ff;font-size:13px;">Challenge ENS Data</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Produits", "84 916")
            st.metric("Weighted F1", "0.8403", "+0.0290 vs benchmark")
        with c2:
            st.metric("Classes", "27")
            st.metric("Benchmark", "0.8113", "Rakuten texte")

    with col_intro:
        st.markdown("## Problématique")
        st.markdown("""
        <div class="info-rakuten">
        Rakuten France met à disposition un catalogue de <strong>84 916 fiches produits</strong> 
        pour lesquelles il faut prédire automatiquement le <strong>code type produit (prdtypecode)</strong> 
        parmi <strong>27 catégories</strong>, à partir du titre, de la description et de l'image de chaque produit.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        Ce problème est fondamental pour tout e-commerce à grande échelle :
        - **Recherche personnalisée** et recommandation de produits
        - **Gestion des doublons** entre produits neufs et d'occasion
        - **Scalabilité** : les approches manuelles ne sont pas viables au-delà de quelques milliers de produits

        La difficulté principale vient du fait que les catalogues modernes contiennent une distribution 
        **fortement déséquilibrée** des catégories et des libellés produits **intrinsèquement bruités**, 
        rédigés par des vendeurs professionnels et non professionnels dans plusieurs langues.
        """)
        st.markdown("## Métrique officielle")
        st.info("**Weighted F1-score** (sklearn, average='weighted') - imposé par le challenge Rakuten / ENS Data. "
                "Cette métrique pénalise les erreurs sur les classes fréquentes plus lourdement que sur les classes rares, "
                "cohérent avec l'objectif de catalogage à grande échelle.")

    st.markdown("---")
    st.markdown("## Stratégie de modélisation")

    col_a, col_b, col_c, col_d = st.columns(4)
    steps = [
        ("1️⃣", "Baseline classique", "TF-IDF + LinearSVC", "Référence rapide et reproductible"),
        ("2️⃣", "Repr. sémantique", "SBERT + MLP", "Embeddings multilingues denses 384-dim"),
        ("3️⃣", "Transfer learning", "ResNet50 Fine-tuning", "Apprentissage progressif en 2 phases"),
        ("4️⃣", "Fusion multimodale", "SBERT + ResNet50 (MLP)", "Late fusion texte + image 2432-dim"),
    ]
    for col, (icon, titre, modele, desc) in zip([col_a, col_b, col_c, col_d], steps):
        with col:
            st.markdown(f"""
            <div style="background:#f0f6ff;border:1px solid #2E75B6;border-radius:8px;
                        padding:15px;text-align:center;height:160px;">
                <div style="font-size:24px;">{icon}</div>
                <strong style="color:#1a2a4a;">{titre}</strong><br>
                <code style="color:#2E75B6;">{modele}</code><br>
                <small style="color:#666;">{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Environnement technique")
    col_env1, col_env2, col_env3 = st.columns(3)
    with col_env1:
        st.markdown("**Matériel**")
        st.markdown("""
        - CPU : Intel Core Ultra 9 275HX
        - RAM : 32 Go DDR5
        - GPU : RTX 5070 Ti Laptop (12 Go VRAM, Blackwell sm_120)
        """)
    with col_env2:
        st.markdown("**Logiciels**")
        st.markdown("""
        - Python 3.10 (conda rakuten_env)
        - PyTorch 2.12.0 nightly cu128
        - CUDA 12.9
        """)
    with col_env3:
        st.markdown("**Contraintes rencontrées**")
        st.markdown("""
        - Architecture GPU Blackwell non supportée par PyTorch stable → nightly requis
        - num_workers=0 sous Windows/Jupyter → cache numpy uint8 (12.8 Go RAM)
        - SHAP MemoryError sur batch → coefficients LinearSVC utilisés
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 2 - EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("Exploration des données")

    df = load_dataset()
    if df is None:
        st.warning("Dataset df_train_processed.csv non trouvé. Vérifiez le chemin : "
                   f"{PATHS['data_proc']}")
        st.info("Les visualisations statiques (figures pré-générées) sont tout de même disponibles ci-dessous.")
    else:
        st.success(f"Dataset chargé : {len(df):,} produits - {df['prdtypecode'].nunique()} classes")

    # ── 2.1 Statistiques clés ─────────────────────────────────────────────
    st.markdown("## Statistiques clés")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Produits (train)", "84 916")
    c2.metric("Classes", "27")
    c3.metric("Ratio max/min classes", "13.4×")
    c4.metric("% NaN description", "35.1 %")
    c5.metric("% images disponibles", "100 %")

    st.markdown("---")

    # ── 2.2 Distribution des classes ─────────────────────────────────────
    st.markdown("## Distribution des classes (prdtypecode)")

    fig_dist = load_figure("01_distribution_classes.png")
    if fig_dist:
        st.image(fig_dist, use_container_width=True)
    else:
        if df is not None:
            fig, ax = plt.subplots(figsize=(14, 5))
            counts = df["prdtypecode"].value_counts().sort_index()
            colors = ["#2E75B6" if v >= counts.mean() else "#a0c4e8" for v in counts.values]
            ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="white")
            ax.set_xlabel("prdtypecode", fontsize=11)
            ax.set_ylabel("Nombre de produits", fontsize=11)
            ax.set_title("Distribution des classes - dataset d'entraînement (84 916 produits)", fontsize=13)
            ax.axhline(counts.mean(), color="red", linestyle="--", linewidth=1, label=f"Moyenne = {counts.mean():.0f}")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("""
    **Constat.** Le dataset contient 27 classes avec un ratio max/min de 13.4 : la classe 2583 (piscines/spas) 
    concentre ~10 000 produits tandis que la classe 1180 n'en compte que ~750. Ce déséquilibre justifie 
    l'utilisation du **weighted F1-score** comme métrique principale, qui pondère chaque classe par son support.
    """)

    st.markdown("---")

    # ── 2.3 Longueur des textes ───────────────────────────────────────────
    col_txt, col_lang = st.columns(2)

    with col_txt:
        st.markdown("## Longueur des textes")
        fig_len = load_figure("03_longueur_textes.png")
        if fig_len:
            st.image(fig_len, use_container_width=True)
        else:
            if df is not None:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                df["n_mots_desig"] = df["designation"].fillna("").apply(lambda x: len(str(x).split()))
                df["n_mots_desc"]  = df["description"].fillna("").apply(lambda x: len(str(x).split()))
                axes[0].hist(df["n_mots_desig"].clip(0, 50), bins=30, color="#2E75B6", edgecolor="white")
                axes[0].set_title("Désignation (mots)")
                axes[1].hist(df["n_mots_desc"].clip(0, 200), bins=30, color="#a0c4e8", edgecolor="white")
                axes[1].set_title("Description (mots)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        st.markdown("""
        Désignations courtes (médiane 11 mots, max 56). Descriptions longues mais optionnelles - 
        35 % manquantes selon les vendeurs. Ces deux champs sont concaténés en `text_combined`.
        """)

    with col_lang:
        st.markdown("## Répartition des langues")
        fig_lang = load_figure("05_repartition_langues.png")
        if fig_lang:
            st.image(fig_lang, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            langs = {"fr": 61, "en": 23, "de": 6, "autres": 10}
            colors = ["#2E75B6", "#4a90d9", "#7fb3e8", "#d5e8f0"]
            ax.pie(langs.values(), labels=[f"{k}\n{v}%" for k, v in langs.items()],
                   colors=colors, startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
            ax.set_title("Langues des désignations")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        st.markdown("""
        61 % de désignations en français, 23 % en anglais, 6 % en allemand. 
        Ce multilinguisme motive le choix de **paraphrase-multilingual-MiniLM-L12-v2** 
        comme modèle d'embedding.
        """)

    st.markdown("---")

    # ── 2.4 Taux de description et images ────────────────────────────────
    col_desc, col_img = st.columns(2)

    with col_desc:
        st.markdown("## Taux de description par classe")
        fig_nanrate = load_figure("05_description_par_classe.png")
        if fig_nanrate:
            st.image(fig_nanrate, use_container_width=True)
        else:
            st.info("Figure 05 non trouvée - exécutez le notebook 01.")
        st.markdown("""
        Le taux de description varie de 10 % à 95 % selon les classes. 
        Les classes techniques (2705, livres) ont quasi-systématiquement une description ; 
        les classes jouets ont souvent des désignations seules.
        """)

    with col_img:
        st.markdown("## Exemples de produits du dataset")
        st.markdown("Exemples de produits issus du dataset d'entraînement :")
        example_imgs = [
            ("image_1000392_product_1257596.jpg",   "Classe : jeux vidéo"),
            ("image_1000076039_product_580161.jpg",  "Classe : livres"),
            ("image_1000093804_product_343306951.jpg","Classe : cartes à collectionner"),
        ]
        cols = st.columns(3)
        for col, (fname, caption) in zip(cols, example_imgs):
            # Cherche d'abord dans image_train, ensuite à côté de app.py (upload projet)
            candidates = [
                PATHS["image_train"] / fname,
                ROOT / fname,
                ROOT.parent / fname,
            ]
            img_found = next((str(p) for p in candidates if p.exists()), None)
            if img_found:
                col.image(img_found, caption=caption, use_container_width=True)
            else:
                col.markdown(
                    f"<div style='border:1px solid #ddd;border-radius:6px;padding:12px;"
                    f"text-align:center;color:#999;font-size:12px;'>"
                    f"Image non trouvée<br><small>{caption}</small></div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.markdown("## Synthèse : implications pour la modélisation")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown("**Texte**")
        st.success("Signal fort. Désignations riches en noms propres et références → TF-IDF avec bigrammes très efficace.")
    with col_s2:
        st.markdown("**Image**")
        st.warning("Signal complémentaire variable. Utile pour les classes ambiguës (livres vs magazines), "
                   "peu discriminant pour jouets/jeux de société.")
    with col_s3:
        st.markdown("**Classes difficiles**")
        st.error("Cluster livre/presse (10, 2280, 2403, 2705) et paire jouets/jeux (1280, 1281) : "
                 "frontières lexicales floues, F1 < 0.70 pour les classes 10 et 1281.")


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 3 - MODÉLISATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("Résultats de modélisation")

    # ── 3.1 Tableau récapitulatif ─────────────────────────────────────────
    st.markdown("## Comparaison globale des modèles")

    results_df = load_results()

    # Affichage du tableau stylisé
    def highlight_best(row):
        if "TF-IDF" in str(row.get("Modele", "")):
            return ["background-color: #d5e8f0; font-weight: bold"] * len(row)
        if "Benchmark" in str(row.get("Modele", "")) or "benchmark" in str(row.get("Type", "")):
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    display_df = pd.DataFrame({
        "Modèle"             : results_df["Modele"] if "Modele" in results_df else results_df.iloc[:, 0],
        "Modalité"           : results_df["Modalite"] if "Modalite" in results_df else "",
        "Weighted F1"        : results_df["Weighted_F1"] if "Weighted_F1" in results_df else results_df.iloc[:, 2],
        "Écart benchmark texte": ""
    })
    bm_texte = 0.8113
    display_df["Écart benchmark texte"] = display_df["Weighted F1"].apply(
        lambda x: f"+{x - bm_texte:.4f}" if (x - bm_texte) > 0 else f"{x - bm_texte:.4f}"
    )
    display_df["Weighted F1"] = display_df["Weighted F1"].apply(lambda x: f"{x:.4f}")

    st.dataframe(
        display_df.style.apply(highlight_best, axis=1),
        use_container_width=True, height=320
    )
    st.caption("Bleu = meilleur modèle | Jaune = benchmarks officiels Rakuten")

    # ── 3.2 Figure comparative ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Figure 12 - Comparaison visuelle")
    fig12 = load_figure("12_comparaison_modeles.png")
    if fig12:
        st.image(fig12, use_container_width=True)
    else:
        st.info("Figure 12 non trouvée dans reports/figures/.")

    st.markdown("""
    Seul le **TF-IDF + LinearSVC** (0.8403) dépasse le benchmark texte Rakuten (0.8113). 
    L'écart de +2.9 points valide l'efficacité des approches classiques de NLP lorsque les données 
    textuelles contiennent des termes hautement spécifiques (noms de produits, marques, références).
    Le benchmark image Rakuten (0.5534) est largement dépassé par tous nos modèles à base d'image.
    """)

    st.markdown("---")

    # ── 3.3 Analyse par modèle ────────────────────────────────────────────
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("## TF-IDF + LinearSVC - Matrice de confusion")
        fig08 = load_figure("08_confusion_matrix_baseline.png")
        if fig08:
            st.image(fig08, use_container_width=True)
        else:
            st.info("Figure 08 non trouvée.")
        st.markdown("""
        **F1 = 0.8403** - 25 classes sur 27 dépassent F1 = 0.70. 
        Les principales confusions concernent le cluster livre/presse 
        (classes 10, 2280, 2403, 2705) et la paire jouets/jeux de société (1280/1281).
        """)

    with col_m2:
        st.markdown("## F1-score par classe - TF-IDF + LinearSVC")
        fig09 = load_figure("09_f1_par_classe_baseline.png")
        if fig09:
            st.image(fig09, use_container_width=True)
        else:
            st.info("Figure 09 non trouvée.")
        st.markdown("""
        Deux classes sous le seuil F1 = 0.70 : **classe 10** (livres, F1 = 0.557) 
        et **classe 1281** (jeux de société, F1 = 0.613). 
        Classes 2905 et 2583 proches de 1.0 grâce à un vocabulaire très spécifique.
        """)

    st.markdown("---")
    col_m3, col_m4 = st.columns(2)

    with col_m3:
        st.markdown("## SBERT + MLP - Courbes d'apprentissage")
        fig10 = load_figure("10_learning_curve_mlp_sbert.png")
        if fig10:
            st.image(fig10, use_container_width=True)
        else:
            st.info("Figure 10 non trouvée.")
        st.markdown("""
        **F1 = 0.7542** après 30 epochs. Convergence stable sans surapprentissage, 
        mais plafond structurel lié à la compression sémantique à 384 dimensions.
        Le modèle reste en deçà du benchmark Rakuten texte (0.8113).
        """)

    with col_m4:
        st.markdown("## ResNet50 - Courbes d'apprentissage (2 phases)")
        fig11 = load_figure("11_learning_curve_resnet50.png")
        if fig11:
            st.image(fig11, use_container_width=True)
        else:
            st.info("Figure 11 non trouvée.")
        st.markdown("""
        **Phase 1** (backbone gelé, 10 epochs) : F1 = 0.6174. 
        **Phase 2** (dégel layer4, 15 epochs) : F1 = **0.6836** (+13 points vs benchmark image). 
        La transition visible à l'epoch 10 montre l'accélération apportée par le fine-tuning.
        """)

    st.markdown("---")
    st.markdown("## Fusion multimodale - Matrice de confusion")
    fig13 = load_figure("13_confusion_matrix_fusion.png")
    if fig13:
        st.image(fig13, use_container_width=True)
    else:
        st.info("Figure 13 non trouvée.")
    st.markdown("""
    **F1 = 0.7105** - La fusion est inférieure au SBERT seul (0.7542). 
    Le vecteur ResNet50 avgpool (2048 dim) domine numériquement le vecteur SBERT (384 dim) 
    dans la concaténation : sans normalisation séparée par modalité, le MLP de fusion 
    pondère implicitement l'image plus fortement, dégradant le signal textuel.  
    La classe 10 (livres) bénéficie toutefois de l'image (+56 rappels corrects vs TF-IDF).
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 4 - CLASSIFICATION EN DIRECT (PoC)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("Classification en direct - Démo PoC")
    st.markdown("""
    <div class="info-rakuten">
    <strong>Proof of Concept</strong> - Ce module démontre la valeur métier du système : 
    saisissez la désignation et la description d'un produit Rakuten, le modèle TF-IDF + LinearSVC 
    prédit instantanément sa catégorie parmi les 27 classes du catalogue.
    </div>
    """, unsafe_allow_html=True)

    pipeline, _model_err = load_svc_model()
    model_loaded = pipeline is not None

    if not model_loaded:
        if _model_err and "introuvable" not in _model_err:
            st.warning(
                f"**Erreur de chargement du modele :** {_model_err}\n\n"
                "**Cause probable :** incompatibilite de version scikit-learn entre "
                "l'environnement d'entrainement et l'environnement actuel.  \n"
                "Re-sauvegardez le pkl depuis `rakuten_env` en relancant la cellule "
                "`joblib.dump` du notebook 02."
            )
        else:
            st.warning(
                f"Modele non trouve a l'emplacement : `{PATHS['svc_model']}`  \n"
                "Verifiez que `tfidf_linearsvc.pkl` est dans `models/baselines/`."
            )

    st.markdown("---")

    # ── Initialisation session_state pour les champs de saisie ───────────────
    if "poc_designation" not in st.session_state:
        st.session_state["poc_designation"] = ""
    if "poc_description" not in st.session_state:
        st.session_state["poc_description"] = ""
    if "poc_reset" not in st.session_state:
        st.session_state["poc_reset"] = False

    # Reset déclenché par le bouton "Nouveau test"
    if st.session_state["poc_reset"]:
        st.session_state["poc_designation"] = ""
        st.session_state["poc_description"] = ""
        st.session_state["poc_reset"] = False
        st.rerun()

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("## Saisie du produit")

        designation = st.text_area(
            "Désignation du produit *",
            value=st.session_state["poc_designation"],
            placeholder="Exemple : Dragon Warrior Monsters Game Boy Color",
            height=80,
            key="input_designation",
            help="Titre court du produit, tel qu'il apparaît dans le catalogue Rakuten."
        )

        description = st.text_area(
            "Description (optionnelle)",
            value=st.session_state["poc_description"],
            placeholder="Exemple : RPG japonais pour Game Boy Color. Version import US. Complet avec boîte et notice.",
            height=120,
            key="input_description",
            help="Description complémentaire. Peut être vide - 35 % des produits n'en ont pas."
        )

        # Sauvegarder la saisie courante dans session_state
        st.session_state["poc_designation"] = designation
        st.session_state["poc_description"] = description

        st.markdown("**Ou essayez un exemple :**")
        examples = {
            "Jeu vidéo Game Boy"     : ("Dragon Warrior Monsters Game Boy Color", "RPG game boy color complete in box US version"),
            "Livre de cuisine"        : ("Spécial Fruits Tartes Verrines Crumbles", "Recettes illustrées pour préparer des desserts aux fruits de saison"),
            "Jeu de société"          : ("Monopoly édition classique Hasbro", "Jeu de plateau familial. 2 à 8 joueurs. A partir de 8 ans."),
            "Piscine gonflable"       : ("Piscine Intex Easy Set 305 cm", "Piscine ronde gonflable 305 x 76 cm avec pompe filtrante incluse"),
            "Magazine presse"         : ("L'Equipe du Lundi 15 mars 2026", "Résultats matchs du week-end Ligue 1 et Champions League"),
            "Carte Pokémon"           : ("Phyllali Niveau 1 PV90 Evolution Evoli", "Carte Pokémon française holo rare 7/111 série XY"),
        }
        ex_choice = st.selectbox("Exemples prédéfinis", ["- sélectionner -"] + list(examples.keys()),
                                 key="poc_example_select")
        if ex_choice != "- sélectionner -":
            designation, description = examples[ex_choice]

        # Bouton Classifier (rouge, pleine largeur)
        predict_btn = st.button("Classifier ce produit", type="primary",
                                 use_container_width=True, disabled=not model_loaded)

        # Bouton Nouveau test - vert clair, pleine largeur, juste sous Classifier
        # Injection CSS ciblée sur la clé btn_nouveau_test
        st.markdown("""
        <style>
        #btn_nouveau_test { margin-top: 4px; }
        #btn_nouveau_test button {
            background-color: #d4edda !important;
            color: #1a5c2a !important;
            border: 1px solid #7bc47f !important;
            font-weight: 500 !important;
            width: 100% !important;
        }
        #btn_nouveau_test button:hover {
            background-color: #b8dfc0 !important;
            border-color: #5aaa60 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        reset_container = st.container()
        with reset_container:
            reset_btn = st.button(
                "Nouveau test",
                use_container_width=True,
                key="btn_nouveau_test",
                help="Vide les champs et remet le formulaire à zéro"
            )
        if reset_btn:
            st.session_state["poc_reset"] = True
            st.rerun()

    with col_result:
        st.markdown("## Résultat")

        if predict_btn and designation.strip():
            text_combined = (designation.strip() + " " + description.strip()).strip()

            try:
                # Prédiction
                prediction = pipeline.predict([text_combined])[0]
                label = LABEL_MAP.get(prediction, f"Classe {prediction}")

                # Score de décision pour la confiance relative
                decision_scores = pipeline.decision_function([text_combined])[0]
                classes = pipeline.classes_
                idx_pred = list(classes).index(prediction)
                score_pred = decision_scores[idx_pred]

                # Top 5 des classes par score de décision
                top5_idx = np.argsort(decision_scores)[::-1][:5]
                top5 = [(classes[i], decision_scores[i], LABEL_MAP.get(classes[i], f"Classe {classes[i]}"))
                        for i in top5_idx]

                # Normalisation softmax pour affichage
                exp_scores = np.exp(decision_scores - decision_scores.max())
                proba_approx = exp_scores / exp_scores.sum()
                conf_pred = proba_approx[idx_pred] * 100

                # Niveau de confiance
                if conf_pred >= 50:
                    conf_class, conf_icon = "confidence-high", "✅"
                elif conf_pred >= 20:
                    conf_class, conf_icon = "confidence-med", "⚠️"
                else:
                    conf_class, conf_icon = "confidence-low", "❌"

                # Affichage résultat principal
                st.markdown(f"""
                <div class="result-box">
                    <p>Catégorie prédite</p>
                    <h2>Code {prediction}</h2>
                    <p style="font-size:18px;color:white;">{label}</p>
                </div>
                """, unsafe_allow_html=True)

                col_conf, col_code = st.columns(2)
                col_conf.metric("Confiance approx.", f"{conf_pred:.1f} %")
                col_code.metric("Score de décision", f"{score_pred:.3f}")

                st.markdown("#### Top 5 des catégories candidates")
                fig_top, ax_top = plt.subplots(figsize=(6, 3))
                top5_labels = [f"{c} - {LABEL_MAP.get(c, c)[:20]}" for c, _, _ in top5]
                top5_scores = [s for _, s, _ in top5]
                colors_bar = ["#2E75B6" if i == 0 else "#a0c4e8" for i in range(len(top5))]
                ax_top.barh(top5_labels[::-1], top5_scores[::-1], color=colors_bar[::-1])
                ax_top.set_xlabel("Score de décision")
                ax_top.set_title("Top 5 classes candidates")
                plt.tight_layout()
                st.pyplot(fig_top)
                plt.close()

                # Tokens déclencheurs
                st.markdown("#### Tokens clés dans la désignation")
                try:
                    vectorizer = pipeline.named_steps.get(
                        "tfidfvectorizer",
                        pipeline.steps[0][1]
                    )
                    clf = pipeline.named_steps.get(
                        "linearsvc",
                        pipeline.steps[-1][1]
                    )
                    X_vec = vectorizer.transform([text_combined])
                    coef_class = clf.coef_[list(clf.classes_).index(prediction)]
                    feature_names = vectorizer.get_feature_names_out()
                    nonzero_idx = X_vec.nonzero()[1]
                    contributions = [(feature_names[i], coef_class[i] * X_vec[0, i])
                                     for i in nonzero_idx]
                    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_tokens = contributions[:8]

                    if top_tokens:
                        fig_tok, ax_tok = plt.subplots(figsize=(6, 3))
                        tok_names  = [t[0] for t in top_tokens]
                        tok_scores = [t[1] for t in top_tokens]
                        tok_colors = ["#2E75B6" if s > 0 else "#d0021b" for s in tok_scores]
                        ax_tok.barh(tok_names[::-1], tok_scores[::-1], color=tok_colors[::-1])
                        ax_tok.axvline(0, color="black", linewidth=0.8)
                        ax_tok.set_xlabel("Contribution (coef × TF-IDF)")
                        ax_tok.set_title(f"Tokens actifs → classe {prediction}")
                        plt.tight_layout()
                        st.pyplot(fig_tok)
                        plt.close()
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

        elif predict_btn and not designation.strip():
            st.warning("La désignation est obligatoire.")
        else:
            st.markdown("""
            <div style="border:2px dashed #2E75B6;border-radius:10px;padding:40px;text-align:center;color:#888;">
                <p style="font-size:40px;">🏷️</p>
                <p>Saisissez une désignation et cliquez sur <strong>Classifier</strong></p>
                <p style="font-size:12px;">Le modèle TF-IDF + LinearSVC (F1 = 0.8403) classifie instantanément<br>
                parmi les 27 catégories du catalogue Rakuten France.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Informations techniques ───────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️  Informations techniques sur le modèle de démo"):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown("**Pipeline**")
            st.code("TfidfVectorizer(\n  ngram_range=(1,2),\n  max_features=200_000,\n  sublinear_tf=True\n)\n+\nLinearSVC(C=1.0)")
        with col_t2:
            st.markdown("**Performance**")
            st.markdown("""
            - Weighted F1 = **0.8403**
            - Entraînement : 15.4 sec (CPU)
            - Inference : ~1 200 prédictions/sec
            - Dépasse benchmark Rakuten : **+2.9 pts**
            """)
        with col_t3:
            st.markdown("**Limites connues**")
            st.markdown("""
            - Classes 10 et 1281 : F1 < 0.70
            - Sensible aux désignations très courtes
            - Pas de prise en compte de l'image
            - Multilinguisme géré par TF-IDF (sans sémantique)
            """)


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 5 - INTERPRÉTABILITÉ
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.title("Interprétabilité - SHAP & Analyse des erreurs")

    st.markdown("""
    <div class="info-rakuten">
    Cette section présente les analyses d'interprétabilité conduites dans le notebook 03, 
    fondées sur les coefficients du LinearSVC (équivalents aux valeurs SHAP pour un modèle linéaire)
    et sur la comparaison croisée des trois modèles.
    </div>
    """, unsafe_allow_html=True)

    # ── 5.1 SHAP - tokens discriminants ──────────────────────────────────
    st.markdown("---")
    st.markdown("## Tokens discriminants par classe (SHAP / coefficients LinearSVC)")

    fig16 = load_figure("16_shap_tokens_par_classe.png")
    if fig16:
        st.image(fig16, use_container_width=True)
    else:
        st.info("Figure 16 non trouvée dans reports/figures/.")

    st.markdown("""
    Les barres **bleues** sont les tokens qui poussent le modèle vers la classe, 
    les barres **rouges** s'y opposent. Pour un modèle linéaire, la valeur SHAP d'un token 
    est exactement le produit de son coefficient TF-IDF par sa valeur TF-IDF - 
    ce calcul est donc exact et sans approximation.

    **Observations clés :**
    - Classe 2583 (piscines) : coefficient `piscine` = +7.20, amplitude nettement supérieure aux autres classes → vocabulaire très spécifique.
    - Classe 2280 (presse) : dominée par des bigrammes de titres (`le du`, `la du`) issus de constructions comme « L'Équipe du... »
    - Classe 1280 vs 1281 : `ne rique` (token rouge pour 1280) correspond au vocabulaire de 1281 → confusion structurelle confirmée.
    """)

    # ── 5.2 SHAP - exemple individuel ────────────────────────────────────
    st.markdown("---")
    st.markdown("## Exemple individuel - Analyse d'une erreur")

    col_shap1, col_shap2 = st.columns([1, 1])
    with col_shap1:
        fig17 = load_figure("17_shap_exemple_individuel.png")
        if fig17:
            st.image(fig17, use_container_width=True)
        else:
            st.info("Figure 17 non trouvée.")

    with col_shap2:
        st.markdown("**Produit analysé :**")
        st.code("Spécial Fruits - Tartes Verrines Crumbles & Cie\n"
                "Classe réelle : 10 (livres grand public)\nClasse prédite : 2280 (presse/magazines)")
        st.markdown("""
        **Mécanisme d'erreur identifié :**

        Les tokens `spe cial` et `cial` (bigrammes issus de la tokenisation de « Spécial ») 
        ont des contributions fortement positives vers la classe 2280. 
        Le modèle associe « Spécial » aux numéros spéciaux de magazines (« Hebdo Spécial », « Édition Spéciale »), 
        ce qui est une association valide dans le corpus de presse, mais incorrecte pour ce titre culinaire.

        Le token `cie` (issu de « & Cie ») s'oppose à la prédiction 2280, 
        car ce suffixe est rare dans les titres de magazines.

        **Levier d'amélioration :** normalisation ou dictionnaire des termes ambigus 
        (`Spécial`, `Edition`, `Collection`) selon leur contexte.
        """)

    # ── 5.3 Matrice de confusion classes difficiles ───────────────────────
    st.markdown("---")
    st.markdown("## Matrice de confusion - Classes avec F1 le plus faible")

    fig15 = load_figure("15_confusion_classes_faibles.png")
    if fig15:
        st.image(fig15, use_container_width=True)
    else:
        st.info("Figure 15 non trouvée.")

    st.markdown("""
    **Patterns d'absorption identifiés :**
    - La classe **2280** (presse) absorbe des exemples de 10 (0.14), 2403 (0.10) et 2705
    - La classe **1280** (jouets) absorbe 21 % des jeux de société (1281)
    - Les classes **1301** et **1940** affichent recall 0.50 - forte ambiguïté avec 1280
    """)

    # ── 5.4 Comparaison F1 par classe ─────────────────────────────────────
    st.markdown("---")
    st.markdown("## Comparaison F1 par classe - 3 modèles")

    fig18 = load_figure("18_f1_par_classe_comparaison.png")
    if fig18:
        st.image(fig18, use_container_width=True)
    else:
        st.info("Figure 18 non trouvée.")

    st.markdown("""
    Deux profils s'opposent :

    | Classe | F1 SVC | F1 Fusion | Interprétation |
    |--------|--------|-----------|----------------|
    | 10 - Livres | 0.557 | 0.661 | Image utile (+0.10) pour distinguer livres/magazines |
    | 2705 - Manuels techniques | 0.706 | 0.873 | Image très discriminante (+0.17) |
    | 1281 - Jeux de société | 0.613 | 0.410 | Image dégrade - visuels jouets/jeux ambigus |
    | 1280 - Jouets | 0.738 | 0.466 | Image dégrade fortement |
    """)

    # ── 5.5 Chevauchement des erreurs ─────────────────────────────────────
    st.markdown("---")
    col_ov1, col_ov2 = st.columns([1, 1])

    with col_ov1:
        st.markdown("## Chevauchement des erreurs entre les 3 modèles")
        fig20 = load_figure("20_chevauchement_erreurs.png")
        if fig20:
            st.image(fig20, use_container_width=True)
        else:
            st.info("Figure 20 non trouvée.")

    with col_ov2:
        st.markdown("## Synthèse de la complémentarité")
        st.markdown("<br>", unsafe_allow_html=True)

        comp_data = {
            "Configuration"    : ["Tous corrects (3 modèles)", "Fusion seul faux",
                                   "Fusion corrige SVC", "SVC corrige Fusion", "Tous faux"],
            "Nb exemples"      : [9852, 2291, 1081, 3269, 1290],
            "Pourcentage"      : ["58.0 %", "13.5 %", "6.4 %", "19.2 %", "7.6 %"],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.markdown("""
        **Conclusion clé :** Le SVC corrige 3× plus d'erreurs de la Fusion que l'inverse 
        (3 269 vs 1 081). La fusion dégrade sur 2 291 exemples que le SVC et SBERT 
        classeraient correctement - signe du déséquilibre numérique non compensé entre modalités.

        **7.6 %** des exemples (1 290) sont mal classés par les 3 modèles : 
        plancher d'erreur structurelle difficile à réduire sans données supplémentaires.
        """)

    # ── 5.6 Images erreurs ────────────────────────────────────────────────
    st.markdown("---")
    col_err1, col_err2 = st.columns(2)

    with col_err1:
        st.markdown("## Classe 10 - Exemples mal classés")
        fig19a = load_figure("19_erreurs_classe_10.png")
        if fig19a:
            st.image(fig19a, use_container_width=True)
        else:
            st.info("Figure 19a non trouvée.")
        st.markdown("Confusions vers 2280 (presse) et 2705 (manuels) "
                    "sur des titres dont la désignation est trop courte ou ambiguë.")

    with col_err2:
        st.markdown("## Classe 1281 - Exemples mal classés")
        fig19b = load_figure("19_erreurs_classe_1281.png")
        if fig19b:
            st.image(fig19b, use_container_width=True)
        else:
            st.info("Figure 19b non trouvée.")
        st.markdown("Confusion quasi-systématique vers la classe 1280 (jouets). "
                    "Visuellement, les produits sont difficiles à distinguer - "
                    "la fusion multimodale ne corrige pas ces cas.")