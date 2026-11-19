# Camera Health Monitor

![Python](https://img.shields.io/badge/Python-3.10-295B68)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-295B68)
![YOLOv8n](https://img.shields.io/badge/YOLOv8n-Ultralytics-DD743D)
![Streamlit](https://img.shields.io/badge/Streamlit-interface-DD743D)
![Statut](https://img.shields.io/badge/statut-prototype-3C7582)

**À partir de quand les images d'une caméra de chantier deviennent-elles trop mauvaises pour faire confiance à la détection ?**

Camera Health Monitor est un prototype personnel. Il mesure la qualité d'une image, détecte les personnes avec un modèle pré-entraîné, puis abîme l'image de façon contrôlée pour observer à quel moment la détection décroche.

---

## Le problème

Sur un chantier, la caméra est fixe, mais pendant des mois la scène change : jour et nuit, météo, poussière, optique salie, ouvriers plus ou moins loin de l'objectif. Le modèle de détection, lui, ne change pas. Ce sont ses images d'entrée qui se dégradent.

La question n'est donc pas « le modèle est-il bon ? », mais « à partir de quand ne peut-on plus lui faire confiance, et pourquoi ? ».

## Ce que fait l'outil

L'application Streamlit comporte trois onglets.

| Onglet | Ce qu'il fait |
|---|---|
| **Qualité de l'image** | Calcule trois indicateurs sans IA (luminosité, netteté, contraste) et rend un verdict motivé : exploitable, limite ou inexploitable. |
| **Détection** | Détecte les personnes avec YOLOv8n. Un curseur règle le seuil de confiance, et un graphique montre ce que ce seuil garde et ce qu'il jette. |
| **Robustesse** | Applique trois dégradations par paliers de 0 à 1 (flou, assombrissement, éloignement), compte les personnes conservées à chaque palier et indique le palier de décrochage. |

## Résultats sur une scène de rue

Image de test : une photo de rue prise à Paris, contenant 18 personnes, dont plusieurs partiellement cachées. Seuil de confiance : 0,25. Seuil de NMS : 0,5. Image normalisée à 1920 pixels.

Sur l'image intacte, le modèle détecte **11 personnes sur 18**, soit un rappel d'environ **0,61**. Toutes les boîtes gardées sont sur de vraies personnes (vérification visuelle).

**Personnes détectées à chaque palier**

| Dégradation | 0 | 0,2 | 0,4 | 0,6 | 0,8 | 1 | Palier de décrochage |
|---|---|---|---|---|---|---|---|
| Flou | 11 | 10 | 10 | 10 | 11 | 11 | Non atteint |
| Assombrissement | 11 | 11 | 10 | 10 | 9 | 8 | 1,0 |
| Éloignement | 11 | 12 | 10 | 10 | **5** | **0** | **0,8** |

Le décrochage est défini comme le premier palier où le nombre de personnes détectées passe sous 80 % de celui de l'image intacte.

## Ce que j'en retiens

1. **La distance est la seule dégradation qui fait vraiment décrocher la détection.** Entre les paliers 0,6 et 0,8, une personne passe d'environ 60 à 40 pixels de haut dans l'image vue par le modèle, et la moitié des détections disparaît.
2. **Aucun indicateur de qualité ne voit l'éloignement.** L'image reste nette, lumineuse et contrastée : seules les personnes rétrécissent. Pour surveiller une caméra, il faut donc mesurer la détection elle-même, pas seulement la qualité de l'image.
3. **Le modèle résiste au flou.** YOLO réduit chaque image à 640 pixels avant de la traiter : sur une image de 1920 pixels, le flou est divisé par 3 avant d'atteindre le modèle.
4. **La netteté mesurée dépend de la luminosité et de la résolution.** Multiplier les pixels par 0,3 divise la variance du laplacien par 11 (0,3² = 0,09). La même photo obtient 517 en 5712 pixels et 2594 une fois réduite à 1920.
5. **Les détections proches du seuil sont instables.** Une personne détectée à 0,24 ou 0,26 passe d'un côté ou de l'autre du seuil pour une variation minuscule. D'où des comptes qui remontent parfois sur une image plus dégradée.
6. **Un détecteur n'est pas stable sur les cas limites quand l'échelle change.** Un enfant à moitié caché est détecté à 0,2 sur l'image intacte, et à 0,43 après une légère réduction de l'image.

## Installation

```bash
git clone https://github.com/Kadiatou-yatt/camera-health-monitor.git
cd camera-health-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sur un Mac Intel, PyTorch s'arrête à une version compilée pour NumPy 1. Si un message signale une incompatibilité avec NumPy 2 :

```bash
pip install "numpy<2" "opencv-python<4.12"
```

Le fichier de poids `yolov8n.pt` (environ 6 Mo) se télécharge automatiquement au premier lancement.

## Utilisation

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur. Importer une image JPEG ou PNG, puis parcourir les trois onglets. Le premier calcul de l'onglet Robustesse lance 18 détections (3 dégradations × 6 paliers) et prend quelques dizaines de secondes ; les suivants sont instantanés grâce au cache.

## Structure du projet

```
camera-health-monitor/
├── app.py                  Interface Streamlit (3 onglets)
├── src/
│   ├── quality.py          Indicateurs de qualité et verdict
│   ├── detection.py        Chargement de YOLO, détection, dessin des boîtes
│   ├── degradations.py     Flou, assombrissement, éloignement par paliers
│   └── metrics.py          IoU entre deux boîtes
├── .streamlit/config.toml  Thème de l'interface
├── images/                 Images de test (exclues du dépôt)
├── requirements.txt
└── README.md
```

Les scripts `check_env.py`, `test_quality.py`, `test_detection.py` et `test_degradations.py` ont servi à vérifier chaque brique en ligne de commande avant de construire l'interface.

## Choix techniques

| Choix | Pourquoi |
|---|---|
| **Indicateurs sans apprentissage** (moyenne, variance du laplacien, écart-type) | Calculables sur un flux entier sans coût notable, et explicables : une image est écartée pour un motif chiffré. |
| **Normalisation à 1920 pixels** à l'import | Sans elle, un même palier de flou ou un même seuil de netteté n'a pas le même sens selon la résolution de la photo. |
| **Verdict sans double comptage** | Assombrir une image fait aussi baisser le contraste et la netteté. Ces deux indicateurs ne sont donc jugés que si la luminosité est correcte. |
| **Détection au seuil plancher de 0,05, puis filtrage** | YOLO ne tourne qu'une fois par image et par palier ; le curseur de seuil filtre ensuite la liste sans relancer le modèle. Résultats mis en cache. |
| **NMS à 0,5** au lieu de 0,7 | Supprime les doublons, deux boîtes sur une même personne. Trop bas, ce seuil fusionnerait deux personnes collées dans une foule. |
| **Éloignement simulé sur fond gris uni** (114, 114, 114) | Réduire puis ré-agrandir l'image ne simule pas l'éloignement. Les bordures répliquées créaient des stries. |
| **Dégradations écrites avec OpenCV** plutôt qu'avec Albumentations | Albumentations sert à l'augmentation aléatoire à l'entraînement ; ici, chaque palier doit être fixé. Son `Downscale` ré-agrandit l'image, ce qui ne simule pas l'éloignement. |
| **Seuil de décrochage à 80 %** | Convention choisie : avec 11 personnes, une seule bascule fait varier le taux de 9 %. 80 % tolère ce bruit et signale la perte de plus d'une personne sur cinq. |

## Limites

- Les résultats reposent sur une seule scène de rue : ils montrent une méthode, pas des seuils généralisables.
- Le décrochage est mesuré par un comptage, sans vérité terrain annotée : une détection en plus n'est pas forcément une vraie personne.
- Le modèle est un détecteur généraliste pré-entraîné sur COCO, pas un modèle métier entraîné sur des images de chantier.
- Les dégradations synthétiques approchent les dégradations réelles sans les reproduire : une vraie scène de nuit ajoute du bruit, du flou et des éclairages ponctuels.
- Les paliers ne forment pas une échelle commune : le palier 0,8 du flou n'est pas « aussi fort » que celui de l'éloignement.
- Les seuils du verdict qualité sont calibrés à la main.

## Pistes d'amélioration

1. Annoter les personnes (CVAT, Label Studio) pour calculer précision, rappel et mAP à chaque palier, au lieu d'un simple comptage.
2. Comparer chaque image à une image de référence de la même caméra plutôt qu'à des seuils fixes.
3. Exprimer les paliers dans une grandeur physique : taille d'une personne en pixels pour l'éloignement, taille du flou rapportée à la hauteur d'une personne.
4. Pour les personnes lointaines, tester la détection par tuiles (SAHI) ou une taille d'entrée de 1280 pixels.
5. Confirmer une détection sur plusieurs images successives, pour une caméra fixe, plutôt que sur une seule.
6. Remplacer l'image de rue par des images réelles de chantier, et le modèle généraliste par un modèle métier.

## Données et RGPD

Les images de test montrent des passants reconnaissables, dont des enfants. Elles ne sont pas versionnées : `images/*.jpg` figure dans le `.gitignore`. Pour reproduire les résultats, utiliser ses propres images.

## Autrice

**Kadiatou Yattassaye**, ingénieure Data & IA (Efrei Paris)
[GitHub](https://github.com/Kadiatou-yatt) · [LinkedIn](https://www.linkedin.com/in/kadiatou-yattassaye/)

Prototype personnel, novembre 2026.
