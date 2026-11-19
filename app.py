import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.degradations import DEGRADATIONS, LEVELS
from src.detection import detect, draw_boxes
from src.quality import compute_quality, quality_verdict

ACCENT = "#DD743D"  
DARK = "#295B68"     

# Couleurs des données (palette validée pour les daltoniens)
KEPT_COLOR, REJECTED_COLOR = "#2a78d6", "#CBD5E1"
LABELS = {"blur": "Flou", "darken": "Assombrissement", "distance": "Éloignement"}
SERIES_COLORS = {"Flou": "#2a78d6", "Assombrissement": "#eb6834", "Éloignement": "#1baf7a"}

# Encart de verdict : (fond, texte)
VERDICT_STYLES = {
    "Exploitable": ("#E7F6EC", "#1E7B34"),
    "Limite": ("#FFF4DB", "#8A5A00"),
    "Inexploitable": ("#FDE8E8", "#B42318"),
    "Non applicable": ("#EEF0F3", "#4B5563"),
}

FLOOR = 0.05          # seuil plancher : on détecte tout au-dessus de 0,05, le curseur filtre ensuite
BREAK_RATE = 0.80     # décrochage : moins de 80 % des personnes de l'image intacte


@st.cache_data(show_spinner=False)
def boxes_at(image, degradation, level):
    # Dégrade l'image puis détecte tout ce qui dépasse le seuil plancher.
    degraded = DEGRADATIONS[degradation](image, level)
    return detect(degraded, conf_min=FLOOR)["boxes"]


def keep(boxes, threshold):
    # Ne garde que les détections dont la confiance atteint le seuil choisi.
    return [b for b in boxes if b["conf"] >= threshold]


def verdict_box(title, verdict):
    bg, fg = VERDICT_STYLES[verdict]
    st.markdown(f'<div class="verdict" style="background:{bg};color:{fg}">'
                f'{title} : {verdict}</div>', unsafe_allow_html=True)


st.set_page_config(page_title="Camera Health Monitor", page_icon="📷", layout="wide")


# Style : bandeau d'en-tête, cartes d'indicateurs, encart de verdict
st.markdown(f"""
<style>
.hero {{
    background: {DARK}; padding: 1.6rem 2rem; border-radius: 12px;
    border-left: 6px solid {ACCENT}; margin-bottom: 1.5rem;
}}
.hero h1 {{ color: white; margin: 0; font-size: 2rem; }}
.hero p {{ color: #D1D5DB; margin: .5rem 0 0 0; font-size: 1.05rem; }}
div[data-testid="stMetric"] {{
    background: #F4F6F8; border-radius: 10px; padding: 1rem;
    border-top: 4px solid {ACCENT};
}}
.verdict {{
    padding: .9rem 1.2rem; border-radius: 10px; font-weight: 600;
    font-size: 1.1rem; margin: 1.2rem 0 .6rem 0;
}}
</style>
""", unsafe_allow_html=True)

# --- En-tête ---
st.markdown("""
<div class="hero">
  <h1>Camera Health Monitor</h1>
  <p>À partir de quand les images d'une caméra de chantier deviennent-elles
  trop mauvaises pour faire confiance à la détection ?</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Le problème et la méthode")
st.markdown("""
Sur un chantier la caméra est fixe, mais pendant des mois les images changent :
nuit, brume, pluie, optique salie, ouvriers plus ou moins loin. Le modèle ne change pas,
ce sont ses entrées qui se dégradent. Sur cette base, on ne se demande plus si le mon modèle qu'on a est-il bon,  mais plutot « à partir de quand je ne peux plus lui faire confiance, et pourquoi ? ».

1. **Qualité** : Mesurer la qualité d'une image avec trois indicateurs calculés sur les pixels, sans IA : luminosité, netteté, contraste. Un verdict est fournit : exploitable, limite, inexploitable.
2. **Détection** : un YOLOv8n pré-entraîné compte les personnes.
3. **Robustesse** : on abîme l'image par paliers de 0 à 1 (flou, assombrissement, éloignement)
   et on mesure à partir de quand la détection décroche.
""")

# --- Barre latérale ---
with st.sidebar:
    st.markdown("**Kadiatou Yattassaye**")
    st.write("Ingénieure Data & IA")
    st.write("Prototype personnel, novembre 2026.")

    st.markdown("[🔗 GitHub](https://github.com/Kadiatou-yatt)")
    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/kadiatou-yattassaye/)")  
# --- Import de l'image ---
st.subheader("Importer une image de caméra")
uploaded = st.file_uploader("Image de caméra",type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Importez une image pour lancer l'analyse.")
    st.stop()  # tant qu'il n'y a pas d'image, on n'exécute pas la suite

# getvalue() : renvoie tout le fichier, même quand Streamlit relance le script
img = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Impossible de lire cette image.")
    st.stop()

# Normalisation : toute image est ramenée à 1920 px maximum.
MAX_SIDE = 1920
original_size = f"{img.shape[1]} × {img.shape[0]} px"
if max(img.shape[:2]) > MAX_SIDE:
    scale = MAX_SIDE / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)

tab1, tab2, tab3 = st.tabs(["Qualité de l'image", "Détection", "Robustesse"])

# --- Onglet 1 : qualité ---
with tab1:
    q = compute_quality(img)
    verdict, reasons = quality_verdict(q)

    col_img, col_kpi = st.columns([3, 2], gap="large")

    with col_img:
        st.image(img, channels="BGR",
                 caption=f"{uploaded.name} · original {original_size}, analysée en {img.shape[1]} × {img.shape[0]} px")

    with col_kpi:
        st.subheader("Indicateurs")
        c1, c2, c3 = st.columns(3)
        c1.metric("Luminosité", f"{q['brightness']:.0f}",
                  help="Moyenne des niveaux de gris, de 0 (noir) à 255 (blanc).")
        c2.metric("Netteté", f"{q['sharpness']:.0f}",
                  help=" Force des contours(Variance du laplacien). Élevée : contours francs. Faible : flou.")
        c3.metric("Contraste", f"{q['contrast']:.0f}",
                  help="Écart-type des niveaux de gris (entre les zones claires et sombres). Faible : image voilée.")

        # Couleur de l'encart selon le verdict : (fond, texte)
        styles = {
            "Exploitable": ("#E7F6EC", "#1E7B34"),
            "Limite": ("#FFF4DB", "#8A5A00"),
            "Inexploitable": ("#FDE8E8", "#B42318"),
        }
        bg, fg = styles[verdict]
        st.markdown(f'<div class="verdict" style="background:{bg};color:{fg}">'
                    f'Verdict : {verdict}</div>', unsafe_allow_html=True)
        for r in reasons:
            st.markdown(f"- {r}")

        with st.expander("Comment lire ces indicateurs"):
            st.markdown("""
**Luminosité (0 à 255)** : à quel point l'image est claire.
Sous 40 : nuit ou scène trop sombre. Au-dessus de 215 : image délavée, surexposée.

**Netteté** : à quel point les contours sont francs. Sous 100 : flou probable
(bougé, optique sale, mise au point perdue). Attention : elle baisse aussi quand
l'image s'assombrit, alors elle n'est jugée que si la luminosité est correcte.

**Contraste** : à quel point les clairs et les sombres se distinguent.
Sous 25 : image voilée, brume, poussière ou buée possible.

Ces seuils sont indicatifs et calibrés à la main. En production, on les recalibrerait
pour chaque caméra, en comparant ses images à une image de référence.
""")

# --- Onglet 2 : détection ---
with tab2:
    threshold = st.slider(
        "Seuil de confiance", 0.05, 0.95, 0.25, 0.05,
        help="Une détection n'est conservée que si sa confiance atteint ce seuil.",
    )

    all_boxes = boxes_at(img, "blur", 0.0)   # palier 0 : l'image intacte
    kept = keep(all_boxes, threshold)
    mean_conf = sum(b["conf"] for b in kept) / len(kept) if kept else 0.0
    fragile = [b for b in kept if b["conf"] < threshold + 0.10]

    col_img, col_kpi = st.columns([3, 2], gap="large")

    with col_img:
        st.image(draw_boxes(img, kept), channels="BGR",
                 caption=f"{len(kept)} personnes détectées au seuil {threshold:.2f}")

    with col_kpi:
        st.subheader("Résultat")
        c1, c2, c3 = st.columns(3)
        c1.metric("Personnes détectées", len(kept))
        c2.metric("Confiance moy.", f"{mean_conf:.2f}",
          help="Moyenne des scores des personnes gardées. Piège : elle monte quand on monte "
               "le seuil, parce qu'on jette les détections faibles, pas parce que le modèle "
               "devient meilleur.")
        c3.metric("Détections fragiles", len(fragile),
                  help="Conservées, mais à moins de 0,10 au-dessus du seuil : "
                       "une petite perturbation peut les faire disparaître.")

        if all_boxes:
            st.markdown("**Ce que le seuil garde, et ce qu'il jette**")
            st.caption(f"{len(all_boxes)} personnes proposées par YOLO, triées de la plus sûre à la moins sûre.")
            confs = sorted((b["conf"] for b in all_boxes), reverse=True)
            df = pd.DataFrame({"Rang": range(1, len(confs) + 1), "Confiance": confs})
            df["Statut"] = np.where(df["Confiance"] >= threshold, "Conservée", "Rejetée")

            bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Rang:O", title=None, axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y("Confiance:Q", scale=alt.Scale(domain=[0, 1]), title="Confiance"),
                color=alt.Color("Statut:N",
                                scale=alt.Scale(domain=["Conservée", "Rejetée"],
                                                range=[KEPT_COLOR, REJECTED_COLOR]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=[alt.Tooltip("Rang:O"), alt.Tooltip("Confiance:Q", format=".2f"),
                         alt.Tooltip("Statut:N")],
            )
            rule = alt.Chart(pd.DataFrame({"Seuil": [threshold]})).mark_rule(
                color=DARK, strokeDash=[4, 4], strokeWidth=2).encode(y="Seuil:Q")
            st.altair_chart(bars + rule)
            st.caption("Hauteur = confiance du modèle.")
            st.caption(" Ligne pointillée = le seuil choisi.")
            st.caption("En bleu, les personnes gardées  et en gris, celles jetées.")
            st.caption("Monter le seuil : des barres passent au gris, et on perd des personnes.")
        else:
            st.warning("Aucune personne détectée sur cette image.")

# --- Onglet 3 : robustesse ---
with tab3:
    st.caption(f"Seuil de confiance utilisé : {threshold:.2f} (réglable dans l'onglet Détection).")

    c_sel, c_lvl = st.columns(2)
    deg = c_sel.selectbox("Dégradation", list(DEGRADATIONS), format_func=lambda k: LABELS[k])
    level = c_lvl.select_slider("Palier", options=LEVELS, value=0.6)

    ref_count = len(keep(boxes_at(img, deg, 0.0), threshold))
    degraded = DEGRADATIONS[deg](img, level)
    kept_deg = keep(boxes_at(img, deg, level), threshold)
    rate = len(kept_deg) / ref_count if ref_count else 0.0
    verdict_deg, _ = quality_verdict(compute_quality(degraded))

    col_img, col_kpi = st.columns([3, 2], gap="large")

    with col_img:
        st.image(draw_boxes(degraded, kept_deg), channels="BGR",
                 caption=f"{LABELS[deg]}, palier {level}")

    with col_kpi:
        st.subheader("À ce palier")
        c1, c2 = st.columns(2)
        c1.metric("Personnes détectées", len(kept_deg), delta=len(kept_deg) - ref_count,
                  help="Écart par rapport à l'image intacte.")
        c2.metric("Taux conservé", f"{rate:.0%}")

        if deg == "distance":
            verdict_box("Verdict qualité", "Non applicable")
            st.caption("L'éloignement ne dégrade pas l'image : il rend les personnes plus petites. "
                    "Les indicateurs de qualité ne peuvent pas le voir.")
            if rate > 1:
                st.info("Plus de personnes que sur l'image intacte : des détections fragiles, "
                        "juste sous le seuil, l'ont franchi. Ce n'est pas une amélioration.")
            elif rate < BREAK_RATE:
                st.warning("La détection a décroché, et aucun indicateur de qualité ne pouvait "
                        "le signaler. Seule la mesure de la détection le révèle.")
        else:
            verdict_box("Verdict qualité", verdict_deg)
            if rate > 1:
                st.info("Plus de personnes que sur l'image intacte : des détections fragiles, "
                        "juste sous le seuil, l'ont franchi. Ce n'est pas une amélioration.")
            elif verdict_deg != "Exploitable" and rate >= BREAK_RATE:
                st.info("L'indicateur de qualité juge l'image dégradée, mais la détection "
                        "tient encore. Les deux mesures ne disent pas la même chose.")
            elif verdict_deg == "Exploitable" and rate < BREAK_RATE:
                st.warning("L'image paraît correcte, mais la détection a décroché. "
                        "La qualité seule ne suffit pas à prédire la détection.")
            elif verdict_deg != "Exploitable" and rate < BREAK_RATE:
                st.error("Les deux mesures concordent : image dégradée et détection décrochée.")
            else:
                st.success("Les deux mesures concordent : image correcte et détection stable.")

    st.divider()
    st.subheader("Courbes de décrochage")

    if ref_count == 0:
        st.warning("Aucune personne détectée sur l'image intacte à ce seuil : "
                   "impossible de mesurer un décrochage.")
    else:
        with st.spinner("Détection sur tous les paliers (lent au premier calcul seulement)..."):
            rows = []
            for name in DEGRADATIONS:
                for lv in LEVELS:
                    n = len(keep(boxes_at(img, name, lv), threshold))
                    rows.append({"Dégradation": LABELS[name], "Palier": lv,
                                 "Personnes": n, "Taux conservé": n / ref_count})
        curves = pd.DataFrame(rows)

        lines = alt.Chart(curves).mark_line(
            strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
            x=alt.X("Palier:Q", scale=alt.Scale(domain=[0, 1]), title="Palier de dégradation"),
            y=alt.Y("Taux conservé:Q", axis=alt.Axis(format="%"),
                    title="Personnes conservées / image intacte"),
            color=alt.Color("Dégradation:N",
                            scale=alt.Scale(domain=list(SERIES_COLORS),
                                            range=list(SERIES_COLORS.values())),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Dégradation:N", "Palier:Q", "Personnes:Q",
                     alt.Tooltip("Taux conservé:Q", format=".0%")],
        )
        break_line = alt.Chart(pd.DataFrame({"y": [BREAK_RATE]})).mark_rule(
            color=DARK, strokeDash=[4, 4]).encode(y="y:Q")
        break_label = alt.Chart(pd.DataFrame(
            {"x": [1.0], "y": [BREAK_RATE], "t": ["Seuil de décrochage : 80 %"]})).mark_text(
            align="right", dy=-8, color=DARK).encode(x="x:Q", y="y:Q", text="t:N")
        st.altair_chart(lines + break_line + break_label)

        # Premier palier où le taux conservé passe sous 80 %
        summary = []
        for name in DEGRADATIONS:
            sub = curves[curves["Dégradation"] == LABELS[name]]
            below = sub[sub["Taux conservé"] < BREAK_RATE]
            summary.append({
                "Dégradation": LABELS[name],
                "Palier de décrochage": str(below["Palier"].iloc[0]) if len(below) else "Non atteint",
            })
        st.dataframe(pd.DataFrame(summary), hide_index=True)

        with st.expander("Voir toutes les données"):
            st.dataframe(curves, hide_index=True)