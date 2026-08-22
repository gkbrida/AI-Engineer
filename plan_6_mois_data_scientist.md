# Plan de montée en compétence — 34 semaines (~7,5 mois)
### De Data Scientist confirmé → Spécialiste LLM / MLOps

**Mise à jour :** 3 semaines de fondamentaux (régression logistique, SVM, arbre de décision) ont été ajoutées en tête de programme, avant le boosting. Le plan est ainsi passé de 26 à 29 semaines. Ces fondamentaux sont volontairement placés avant XGBoost/LightGBM/CatBoost : comprendre en profondeur l'arbre de décision est un prérequis pour vraiment maîtriser le boosting plutôt que de l'utiliser comme boîte noire.

**Nouvelle mise à jour :** 5 semaines supplémentaires ont été ajoutées. Une semaine de **bagging** (Random Forest) est insérée juste avant le boosting : les deux méthodes partent du même constat (un arbre seul est instable) mais le résolvent à l'opposé — réduction de variance en parallèle vs réduction de biais en séquentiel — et il est plus pédagogique de voir la version "parallèle et simple" avant la version "séquentielle et sophistiquée". Quatre semaines sont également ajoutées juste après SHAP/LIME : les **métriques d'évaluation** (classification et régression), le **clustering** en deux temps (K-Means/hiérarchique puis DBSCAN/GMM/évaluation), et la **gestion du déséquilibre des classes**. Ces briques sont volontairement placées à cet endroit : elles sont transverses (utiles au reste du programme, y compris en Deep Learning et LLM) et s'appuient directement sur des notions déjà vues (matrice de confusion pour le déséquilibre, arbres pour comprendre pourquoi le clustering ne les utilise pas). Le plan passe donc de 29 à **34 semaines**.

**Format vidéo hebdo (sans visage)** : privilégier l'écran-cast (code, notebook, terminal) commenté en voix off, ou des slides/diagrammes animés (Excalidraw, Canva, PowerPoint) + voix off. Outil recommandé : OBS Studio (gratuit) pour l'enregistrement d'écran.

---

## PHASE 1 — Fondamentaux ML : classification, ensembles, évaluation, clustering & déséquilibre (Semaines 1-11)

### Semaine 1 — Régression logistique : théorie, paramètres, exemples

**À apprendre en détail :**
- Différence avec la régression linéaire : on ne prédit pas une valeur continue mais une **probabilité** d'appartenance à une classe (entre 0 et 1)
- La fonction sigmoïde : `σ(z) = 1 / (1 + e^(-z))`, qui transforme n'importe quelle valeur réelle `z = β₀ + β₁x₁ + ... + βₙxₙ` en une probabilité
- Notion de **logit** (log-odds) : `logit(p) = ln(p / (1-p)) = z` — la régression logistique est en réalité une régression **linéaire sur le logit** de la probabilité, ce qui explique pourquoi la frontière de décision reste linéaire
- Interprétation des coefficients : `exp(β_i)` = **odds ratio** — à combien se multiplie le rapport de cotes (odds) quand la feature augmente d'une unité (ex : `exp(β)=1.5` → chaque unité supplémentaire multiplie les chances par 1,5)
- Fonction de perte : **log-loss / binary cross-entropy**, dérivée du principe du maximum de vraisemblance (pas de solution analytique fermée contrairement à la régression linéaire → estimation itérative par descente de gradient ou Newton-Raphson/IRLS)
- Frontière de décision : toujours **linéaire** dans l'espace des features (un hyperplan) — limite importante à connaître avant d'attaquer les SVM la semaine prochaine
- Seuil de décision (`threshold`, par défaut 0.5) : ajustable selon le contexte métier (ex : abaisser le seuil pour maximiser le rappel en détection de fraude)

**Hyperparamètres clés (sklearn `LogisticRegression`) :**

| Paramètre | Rôle |
|---|---|
| `C` | Inverse de la force de régularisation (attention : petit `C` = régularisation forte, contrairement à `alpha` ailleurs) |
| `penalty` | Type de régularisation : `l1` (Lasso, favorise la parcimonie/sélection de variables), `l2` (Ridge, par défaut), `elasticnet` |
| `solver` | Algorithme d'optimisation (`lbfgs`, `liblinear`, `saga`...) — `liblinear` nécessaire pour `l1` sur petits datasets |
| `class_weight` | Pondération des classes (`balanced` utile en cas de déséquilibre, ex : fraude) |
| `max_iter` | Nombre d'itérations de l'optimiseur (augmenter si le modèle ne converge pas) |

**Exemple pratique :**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_classification(n_samples=500, n_features=4, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# La régularisation nécessite des features standardisées
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(C=1.0, penalty="l2")
model.fit(X_train_scaled, y_train)

# Interprétation des coefficients en odds ratio
odds_ratios = np.exp(model.coef_[0])
for i, odr in enumerate(odds_ratios):
    print(f"Feature {i}: odds ratio = {odr:.2f}")

print("Probabilités prédites (5 premières):", model.predict_proba(X_test_scaled)[:5, 1])
```

**Exercice :** faire varier `C` (0.01, 1, 100) et observer l'effet sur la magnitude des coefficients (plus `C` est petit, plus les coefficients sont écrasés vers 0). Tracer la frontière de décision sur un jeu de données 2D (`make_classification` avec `n_features=2`) pour visualiser sa linéarité.

**🎥 Vidéo :** "La régression logistique, enfin comprise (sigmoïde, odds ratio, régularisation)" — schéma de la sigmoïde + démo code + lecture des odds ratios sur un exemple concret (ex : probabilité de désabonnement client).

---

### Semaine 2 — SVM (Support Vector Machines) : théorie, paramètres, exemples

**À apprendre en détail :**
- Principe central : trouver l'**hyperplan séparateur qui maximise la marge** entre les deux classes (pas juste "un" hyperplan qui sépare, mais celui qui laisse le plus d'espace de chaque côté)
- Notions clés : **hyperplan** (`w·x + b = 0`), **marge** (distance entre l'hyperplan et les points les plus proches de chaque classe, `= 2/||w||`), **vecteurs de support** (les points les plus proches de la frontière, seuls à influencer réellement la position de l'hyperplan)
- Formulation d'optimisation (marge dure) : minimiser `0.5 * ||w||²` sous contrainte `y_i(w·x_i + b) ≥ 1` pour tous les points
- **Marge souple (soft margin)** : ajout de variables d'écart `ξ_i` pour tolérer des erreurs quand les classes ne sont pas parfaitement séparables — le paramètre `C` contrôle l'arbitrage entre marge large et erreurs tolérées
- **Le kernel trick** — la notion la plus importante de la semaine : au lieu de transformer explicitement les données vers un espace de dimension supérieure où elles deviennent séparables linéairement (coûteux), on utilise une **fonction noyau** `K(x_i, x_j)` qui calcule directement le produit scalaire dans cet espace, sans jamais calculer la transformation explicitement
- Noyaux courants :
  - **Linéaire** : `K(x_i,x_j) = x_i · x_j` (pas de transformation, frontière linéaire)
  - **RBF / gaussien** (le plus utilisé) : `K(x_i,x_j) = exp(-γ ||x_i - x_j||²)` — crée des frontières très flexibles
  - **Polynomial** : `K(x_i,x_j) = (x_i · x_j + c)^d`
- Le paramètre `gamma` (noyau RBF) : contrôle la "portée" d'influence d'un point — `gamma` élevé = influence très locale = frontière complexe et sinueuse (risque d'overfitting) ; `gamma` faible = influence large = frontière plus lisse

**Hyperparamètres clés (sklearn `SVC`) :**

| Paramètre | Rôle | Effet si trop élevé |
|---|---|---|
| `C` | Tolérance aux erreurs de classification | Marge étroite, overfitting |
| `kernel` | Type de noyau (`linear`, `rbf`, `poly`, `sigmoid`) | — |
| `gamma` | Portée d'influence d'un point (noyaux non-linéaires) | Frontière très irrégulière, overfitting |
| `degree` | Degré du polynôme (si `kernel="poly"`) | Complexité croissante |

**Exemple pratique :**
```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset non-linéairement séparable
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Comparer noyau linéaire vs RBF
for kernel in ["linear", "rbf"]:
    model = SVC(kernel=kernel, C=1.0, gamma="scale")
    model.fit(X_train, y_train)
    print(f"Kernel={kernel} | Accuracy test: {model.score(X_test, y_test):.3f}")
    print(f"Nombre de vecteurs de support: {len(model.support_)}")
```

**Exercice :** sur `make_moons` (données en forme de croissants), le noyau linéaire devrait clairement sous-performer face au RBF — bonne démonstration visuelle du kernel trick. Faire varier `gamma` (0.1, 1, 10) avec `kernel="rbf"` fixe, visualiser la frontière de décision à chaque fois, et observer le sur-apprentissage progressif. **Rappel important :** le scaling des features (`StandardScaler`) est indispensable pour les SVM — contrairement aux arbres, les distances utilisées dans le kernel sont très sensibles à l'échelle des variables.

**🎥 Vidéo :** "Le kernel trick des SVM, expliqué visuellement" — animation montrant des points non séparables en 2D qui deviennent séparables une fois "projetés" dans un espace supérieur, puis démo du changement de frontière selon `gamma`.

---

### Semaine 3 — L'arbre de décision en détail (la brique fondamentale du bagging et du boosting)

**Pourquoi cette semaine est essentielle :** tous les hyperparamètres et notions vus en semaines 4, 5 et 6 (Bagging, Boosting, XGBoost, LightGBM, CatBoost) sont des **extensions directes** de ce que vous allez apprendre ici. Un arbre de décision mal compris = un bagging ou un boosting utilisé comme boîte noire.

**a) Anatomie d'un arbre**
- **Nœud racine** (*root node*) : le tout premier nœud, contient l'ensemble des données
- **Nœuds internes / nœuds de décision** : chaque nœud interne applique un test sur une feature (ex : `age < 30 ?`) et redirige vers un enfant selon la réponse
- **Branches** : les connexions entre nœuds, représentant le résultat du test (vrai/faux)
- **Feuilles** (*leaf nodes* / *terminal nodes*) : nœuds finaux qui ne se divisent plus, contiennent la prédiction (une classe en classification, une valeur moyenne en régression)
- **Profondeur** (*depth*) : nombre de niveaux entre la racine et la feuille la plus éloignée

**b) Comment un split est choisi (le cœur de l'algorithme CART)**
Pour chaque nœud, l'algorithme teste **toutes les features** et **tous les seuils possibles**, et choisit le split qui **réduit le plus l'impureté** du nœud. C'est une approche **gloutonne (greedy)** : on choisit le meilleur split immédiat à chaque étape, sans anticiper les divisions futures (ce qui explique pourquoi un arbre seul n'est pas toujours optimal — d'où l'intérêt du bagging/boosting pour combiner plusieurs arbres).

**c) Mesures d'impureté (classification)**
- **Indice de Gini** : `Gini = 1 - Σ p_k²`, où `p_k` = proportion de la classe `k` dans le nœud. Un nœud pur (une seule classe présente) a un Gini de 0 ; un nœud avec un mélange 50/50 (cas binaire) a un Gini de 0.5 (impureté maximale)
- **Entropie** : `Entropie = - Σ p_k * log2(p_k)` — mesure issue de la théorie de l'information, interprétation similaire au Gini mais échelle logarithmique
- **Gain d'information** (avec l'entropie) : `Gain = Entropie(parent) - Σ (n_enfant/n_parent) * Entropie(enfant)` — le split choisi est celui qui **maximise ce gain**
- En pratique, Gini et Entropie donnent des résultats très proches ; Gini est légèrement plus rapide à calculer (pas de logarithme) donc souvent le choix par défaut

**d) Mesure pour la régression**
- **Réduction de variance (MSE)** : le split choisi minimise la somme pondérée des variances (ou du MSE) dans les deux nœuds enfants — c'est directement ce qui est utilisé pour entraîner les arbres sur les pseudo-résidus en Gradient Boosting (semaine 5)

**e) Élagage (pruning) — comment éviter l'overfitting**
- **Pré-élagage (pre-pruning)** : contraintes imposées *avant/pendant* la construction (limiter `max_depth`, exiger un nombre minimum d'observations par split ou par feuille)
- **Post-élagage (post-pruning)** : on laisse l'arbre grandir complètement, puis on retire les branches qui n'apportent pas assez de valeur — méthode **cost-complexity pruning** (`ccp_alpha` dans sklearn), qui pénalise chaque feuille supplémentaire d'un coût `α`, exactement comme le `γ` (gamma) vu dans la fonction objectif de XGBoost en semaine 6 !

**f) Notion de pureté d'une feuille**
Une feuille est dite **pure** quand toutes les observations qu'elle contient appartiennent à la même classe (Gini = 0). Un arbre non contraint (sans `max_depth`) grandira jusqu'à ce que toutes les feuilles soient pures, ce qui mène quasi systématiquement à l'**overfitting** (l'arbre mémorise le bruit d'entraînement).

**g) Le lien direct avec bagging et boosting (à bien faire le pont) :**
- **Bagging (Random Forest)** : construit de nombreux arbres **profonds** (peu élagués) sur des échantillons bootstrap différents, ET tire aléatoirement un sous-ensemble de features à chaque split (`max_features`) — la profondeur importe peu individuellement car la moyenne des arbres réduit la variance
- **Boosting** : utilise au contraire des arbres **volontairement peu profonds** (souvent `max_depth=3` à `6`, parfois de simples "stumps" à 1 niveau) comme apprenants faibles — c'est le fait de les enchaîner sur les résidus qui construit la puissance du modèle, pas la complexité individuelle de chaque arbre
- Le `γ` (gain minimum pour split) de XGBoost = exactement le principe de cost-complexity pruning vu ici
- Le `min_child_weight` de XGBoost / `min_samples_leaf` de sklearn = la même contrainte de taille minimale de feuille

**Hyperparamètres clés (sklearn `DecisionTreeClassifier` / `Regressor`) :**

| Paramètre | Rôle |
|---|---|
| `criterion` | Mesure d'impureté : `gini`, `entropy` (classification) ou `squared_error` (régression) |
| `max_depth` | Profondeur maximale de l'arbre — le levier anti-overfitting le plus direct |
| `min_samples_split` | Nombre minimum d'observations requises pour tenter un split |
| `min_samples_leaf` | Nombre minimum d'observations requises dans une feuille |
| `max_features` | Nombre de features considérées à chaque split (pertinent surtout en contexte Random Forest) |
| `ccp_alpha` | Paramètre de post-élagage (cost-complexity pruning) |

**Exemple pratique :**
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Calcul manuel du Gini pour ancrer la formule
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return 1 - np.sum(proportions ** 2)

y_pur = np.array([0, 0, 0, 0])
y_mix = np.array([0, 0, 1, 1])
print(f"Gini nœud pur: {gini(y_pur):.3f}")   # attendu: 0.0
print(f"Gini nœud mixte 50/50: {gini(y_mix):.3f}")  # attendu: 0.5

# 2. Entraîner et visualiser un vrai arbre
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=data.feature_names, class_names=data.target_names,
          filled=True, rounded=True, fontsize=8)
plt.savefig("arbre_decision.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Accuracy test (max_depth=3): {tree.score(X_test, y_test):.3f}")

# 3. Observer l'overfitting en fonction de max_depth
for depth in [1, 2, 3, 5, 10, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    t.fit(X_train, y_train)
    print(f"max_depth={depth} | train: {t.score(X_train, y_train):.3f} | test: {t.score(X_test, y_test):.3f}")
```

**Ce qu'il faut observer :**
1. Le calcul manuel du Gini doit donner 0.0 pour le nœud pur et 0.5 pour le nœud à 50/50 — vérifiez que vous savez refaire ce calcul à la main sur papier avant de continuer
2. En ouvrant `arbre_decision.png`, identifiez visuellement la racine, un nœud interne et une feuille, et lisez la valeur de Gini affichée à chaque nœud
3. Sur la boucle `max_depth`, vous devriez voir l'accuracy train grimper vers 1.0 (overfitting total sans limite de profondeur) pendant que l'accuracy test plafonne puis stagne ou baisse — **c'est précisément le phénomène que le bagging et le boosting sont conçus pour corriger**, chacun à sa manière

**🎥 Vidéo :** "L'arbre de décision : la brique que XGBoost et Random Forest cachent" — schéma annoté d'un arbre (racine/nœuds/feuilles), calcul du Gini en direct sur un petit exemple à la main, puis démo de l'overfitting en faisant varier `max_depth` à l'écran. Terminer en faisant explicitement le pont : "voilà pourquoi la semaine prochaine, le Bagging combine des arbres volontairement laissés profonds — avant que le Boosting, la semaine suivante, inverse complètement la logique avec des arbres peu profonds."

---

### Semaine 4 — Bagging & Random Forest : réduire la variance en parallèle

**À apprendre en détail :**
- **Bagging (Bootstrap Aggregating)** : principe général — entraîner plusieurs modèles **indépendants** sur des tirages différents des données, puis agréger leurs prédictions (vote majoritaire en classification, moyenne en régression) pour réduire la **variance** sans toucher au biais
- **Bootstrap** : tirage aléatoire **avec remise** d'un échantillon de même taille `n` que le jeu d'entraînement original. Un même exemple peut être tiré plusieurs fois, d'autres jamais
- **Calcul de la fraction d'exemples "out-of-bag" (OOB)** : la probabilité qu'une observation donnée ne soit *jamais* tirée sur `n` tirages avec remise est `(1 - 1/n)^n`. Quand `n → ∞`, cette quantité tend vers `1/e ≈ 0.368` — donc en moyenne **63,2 % des observations uniques** se retrouvent dans chaque échantillon bootstrap, et **36,8 % sont laissées de côté** (les OOB) pour cet arbre-là
- **OOB score** : chaque observation n'ayant pas participé à l'entraînement d'un arbre peut servir à l'évaluer — en moyennant ces évaluations sur tous les arbres, on obtient une estimation de la performance en généralisation **sans avoir besoin d'un jeu de test séparé** (une sorte de validation croisée "gratuite")
- **Random Forest = Bagging + tirage aléatoire de features** : à chaque split, seul un sous-ensemble aléatoire des features (`max_features`) est proposé à l'arbre — cela **décorrèle** les arbres entre eux (sans ce tirage, tous les arbres bootstrap auraient tendance à choisir le même split "dominant" en tête d'arbre), ce qui réduit encore davantage la variance de la moyenne
- **Pourquoi ça marche (intuition mathématique)** : la variance d'une moyenne de `B` variables identiquement distribuées mais corrélées avec corrélation `ρ` et variance individuelle `σ²` est `ρσ² + (1-ρ)σ²/B`. Le second terme diminue avec `B`, mais le premier terme (`ρσ²`) est un plancher incompressible tant que `ρ > 0` — d'où l'intérêt de décorréler les arbres via `max_features` pour faire baisser ce plancher, pas seulement d'augmenter `B`
- **Pourquoi des arbres profonds et non élagués** : contrairement au boosting (semaine 5), le bagging a besoin d'apprenants à **faible biais et forte variance** (des arbres profonds, proches de l'overfitting individuel) — c'est justement cette variance que l'agrégation va lisser. Un arbre peu profond (fort biais) n'apporterait rien de plus une fois moyenné
- **Feature importance** : importance par **réduction moyenne d'impureté** (MDI, rapide mais biaisée en faveur des variables à forte cardinalité) vs **importance par permutation** (on mesure la chute de score quand on mélange aléatoirement une colonne — plus lent mais plus fiable)
- **Lien avec la semaine 3** : le bagging est la réponse directe au constat de la semaine dernière — un arbre non contraint overfit (variance élevée). Plutôt que de contraindre l'arbre (`max_depth`), on le laisse overfitter individuellement et on corrige en moyennant plusieurs versions

**Hyperparamètres clés (sklearn `RandomForestClassifier` / `RandomForestRegressor`) :**

| Paramètre | Rôle | Effet si trop élevé / trop bas |
|---|---|---|
| `n_estimators` | Nombre d'arbres dans la forêt | Trop bas : variance encore élevée, estimation instable. Trop haut : coût de calcul inutile (le gain plafonne, pas d'overfitting supplémentaire) |
| `max_features` | Nombre de features tirées aléatoirement à chaque split | Trop haut (proche du nombre total) : arbres corrélés, gain de variance limité. Trop bas : arbres trop faibles individuellement, biais qui augmente |
| `max_depth` | Profondeur max de chaque arbre | Généralement laissé `None` (arbres profonds) — contrairement au boosting |
| `min_samples_leaf` | Taille minimale d'une feuille | Trop bas : arbres très bruités. Trop haut : perte de la variance individuelle utile à l'agrégation |
| `bootstrap` | Active le tirage bootstrap | Si `False` : tous les arbres voient les mêmes données, seul `max_features` les décorrèle |
| `oob_score` | Calcule le score OOB pendant l'entraînement | — |

**Exemple pratique :**
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 1. Vérifier la formule théorique de la fraction OOB (~36,8%)
n = 1000
rng = np.random.default_rng(42)
indices = np.arange(n)
bootstrap_sample = rng.choice(indices, size=n, replace=True)
fraction_oob = 1 - len(np.unique(bootstrap_sample)) / n
print(f"Fraction OOB observée: {fraction_oob:.3f} (théorique: {1/np.e:.3f})")

# 2. Comparer un arbre seul (profond, non élagué) à une Random Forest
tree = DecisionTreeClassifier(max_depth=None, random_state=42)
tree.fit(X_train, y_train)
print(f"Arbre seul       | train: {tree.score(X_train, y_train):.3f} | test: {tree.score(X_test, y_test):.3f}")

rf = RandomForestClassifier(
    n_estimators=200, max_features="sqrt", oob_score=True, random_state=42
)
rf.fit(X_train, y_train)
print(f"Random Forest     | train: {rf.score(X_train, y_train):.3f} | test: {rf.score(X_test, y_test):.3f}")
print(f"Score OOB (estimé sans jeu de test): {rf.oob_score_:.3f}")

# 3. Feature importance : MDI vs permutation
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
for name, mdi, permimp in sorted(
    zip(data.feature_names, rf.feature_importances_, perm.importances_mean),
    key=lambda t: -t[2]
)[:5]:
    print(f"{name:25s} | MDI: {mdi:.3f} | Permutation: {permimp:.3f}")
```

**Exercice :** faire varier `n_estimators` (1, 10, 50, 200) et tracer l'accuracy test — observer le plateau caractéristique du bagging (contrairement au boosting, augmenter `n_estimators` ne fait jamais overfitter). Comparer ensuite le score OOB au score sur un vrai jeu de test mis de côté : ils doivent être très proches, ce qui valide l'intérêt du OOB comme validation "gratuite".

**🎥 Vidéo :** "Bagging & Random Forest : comment 100 arbres imparfaits font un excellent modèle" — schéma du tirage bootstrap, calcul en direct de la fraction OOB (36,8%), puis démo code montrant le plateau de performance quand `n_estimators` augmente. Terminer sur le pont : "le bagging moyenne des arbres indépendants pour réduire la variance — la semaine prochaine, le Boosting va au contraire enchaîner les arbres pour réduire le biais."

---

### Semaine 5 — Boosting : la théorie derrière l'algorithme

**À apprendre en détail :**
- Principe du gradient boosting : ajustement séquentiel sur les résidus (pseudo-résidus = gradient de la loss)
- Différence fondamentale avec le bagging de la semaine 4 (réduction de variance, en parallèle) vs le boosting (réduction de biais, en séquentiel)
- Rôle du *learning rate* (shrinkage) et du nombre d'estimateurs
- Profondeur des arbres faibles (*weak learners*) et compromis biais/variance — lien direct avec la semaine 3 (et contraste avec la semaine 4 : ici les arbres restent volontairement peu profonds)
- AdaBoost vs Gradient Boosting (pondération des erreurs vs descente de gradient)

**Exercice pratique :** implémenter un Gradient Boosting simplifié "from scratch" en Python (sur données jouet) pour bien saisir la logique itérative.

**🎥 Vidéo :** "Comment fonctionne le Boosting, expliqué avec un diagramme animé" — schéma de la construction séquentielle d'arbres.

---

### Semaine 6 — XGBoost, LightGBM, CatBoost en profondeur

**À apprendre en détail :**
- XGBoost : régularisation L1/L2 intégrée à la fonction objectif, approximation de second ordre (Hessienne)
- LightGBM : *histogram-based splitting*, croissance *leaf-wise* vs *level-wise* (plus rapide, risque d'overfitting)
- CatBoost : gestion native des variables catégorielles (*ordered boosting*)
- Hyperparamètres critiques : `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, `early_stopping_rounds`
- Gestion du déséquilibre de classes avec ces librairies (`scale_pos_weight`)

**Exercice pratique :** sur un même dataset (Kaggle), comparer XGBoost / LightGBM / CatBoost en temps d'entraînement, score et stabilité.

**🎥 Vidéo :** "XGBoost vs LightGBM vs CatBoost : lequel choisir ?" — benchmark commenté à l'écran (notebook + tableau de résultats).

---

### Semaine 7 — Interprétabilité des modèles (SHAP/LIME)

**À apprendre en détail :**
- Fondement théorique de SHAP (valeurs de Shapley, théorie des jeux coopératifs)
- Lecture des graphiques : *summary plot*, *force plot*, *dependence plot*, *waterfall plot*
- LIME : approximation locale linéaire autour d'une prédiction
- Différence interprétabilité globale (feature importance) vs locale (SHAP/LIME par prédiction)
- Cas d'usage métier : justifier une décision de crédit, détection de biais

**Exercice pratique :** appliquer SHAP sur le modèle XGBoost entraîné en semaine 6, produire les visualisations et interpréter 3 prédictions individuelles.

**🎥 Vidéo :** "Rendre un modèle de ML explicable avec SHAP" — démonstration à l'écran des graphiques et de leur lecture.

---

### Semaine 8 — Métriques d'évaluation : bien mesurer avant d'optimiser

**Pourquoi cette semaine est essentielle :** jusqu'ici, l'accuracy a servi de repère implicite. Or elle devient trompeuse dès que les classes sont déséquilibrées (semaine 11) ou que le coût d'une erreur diffère selon son type (ex : en scoring de crédit, refuser un bon client ne coûte pas la même chose qu'accepter un mauvais payeur). Cette semaine pose le vocabulaire et les outils utilisés dans tout le reste du programme.

**À apprendre en détail :**
- **Matrice de confusion** : Vrais Positifs (VP), Faux Positifs (FP), Vrais Négatifs (VN), Faux Négatifs (FN) — base de toutes les métriques de classification
- **Accuracy** : `(VP+VN)/(VP+VN+FP+FN)` — et pourquoi elle ment sur données déséquilibrées (un modèle qui prédit toujours "non-fraude" a 99% d'accuracy sur un dataset à 1% de fraude, tout en étant inutile)
- **Précision** : `VP/(VP+FP)` — parmi les prédictions positives, combien sont correctes ? Critique quand un FP coûte cher (ex : bloquer une transaction légitime)
- **Rappel (recall / sensibilité)** : `VP/(VP+FN)` — parmi les vrais positifs, combien sont détectés ? Critique quand un FN coûte cher (ex : laisser passer une fraude, rater un cancer)
- **F1-score** : moyenne **harmonique** de précision et rappel, `F1 = 2·(P·R)/(P+R)` — la moyenne harmonique (contrairement à l'arithmétique) pénalise fortement un déséquilibre entre les deux, donc un modèle ne peut pas "tricher" en sacrifiant l'un pour gonfler l'autre
- **Courbe ROC et AUC** : trace le taux de vrais positifs (rappel) contre le taux de faux positifs (`FP/(FP+VN)`) à tous les seuils de décision possibles. L'AUC s'interprète comme la probabilité que le modèle attribue un score plus élevé à un positif tiré au hasard qu'à un négatif tiré au hasard
- **Courbe Précision-Rappel et PR-AUC** : plus informative que la ROC quand les classes sont très déséquilibrées (la ROC peut sembler excellente même avec un modèle médiocre sur la classe minoritaire, car le taux de FP reste petit en valeur relative face à la masse de négatifs) — lien direct avec la semaine 11
- **Log-loss (cross-entropy)** : `-1/n · Σ [y·log(p) + (1-y)·log(1-p)]` — contrairement à l'accuracy, pénalise une prédiction confiante et fausse bien plus qu'une prédiction hésitante et fausse ; c'est une métrique dite "propre" (proper scoring rule), directement liée à la fonction de perte de la régression logistique (semaine 1)
- **Métriques de régression** : MAE (erreur absolue moyenne, robuste aux outliers), MSE/RMSE (pénalise davantage les grosses erreurs à cause du carré), R² (part de variance expliquée, `1 - SS_res/SS_tot`) et R² ajusté (pénalise l'ajout de variables inutiles)
- **Stratégies de validation croisée** : k-fold classique, **Stratified K-Fold** (préserve la proportion des classes dans chaque pli — indispensable en classification déséquilibrée), **Time Series Split** (jamais de mélange aléatoire sur des données temporelles, chaque pli d'entraînement précède son pli de test dans le temps), Group K-Fold (évite les fuites quand plusieurs lignes appartiennent au même individu/groupe)

**Quelle métrique choisir (repère métier) :**

| Métrique | Formule clé | Cas d'usage typique |
|---|---|---|
| Précision | VP/(VP+FP) | Coût élevé d'un faux positif (ex : blocage de transaction) |
| Rappel | VP/(VP+FN) | Coût élevé d'un faux négatif (ex : fraude non détectée, désabonnement non anticipé) |
| F1-score | 2PR/(P+R) | Besoin d'un compromis unique précision/rappel |
| ROC-AUC | aire sous la courbe TPR/FPR | Classes globalement équilibrées |
| PR-AUC | aire sous la courbe P/R | Classes déséquilibrées (fraude, churn rare) |
| RMSE | racine du MSE | Régression, pénaliser les grosses erreurs (ex : prévision de revenus) |
| MAE | erreur absolue moyenne | Régression, robustesse aux valeurs extrêmes |

**Exemple pratique :**
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, classification_report
)

# Dataset volontairement déséquilibré (5% de positifs, ex : churn rare)
X, y = make_classification(
    n_samples=2000, n_features=10, weights=[0.95, 0.05], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Matrice de confusion:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy trompeuse si on prédit tout en négatif: {(y_test == 0).mean():.3f}")
print(f"Précision: {precision_score(y_test, y_pred):.3f} | Rappel: {recall_score(y_test, y_pred):.3f}")
print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f} | PR-AUC: {average_precision_score(y_test, y_proba):.3f}")
print(f"Log-loss: {log_loss(y_test, y_proba):.3f}")

# Validation croisée stratifiée
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
print(f"F1 en validation croisée stratifiée (5 plis): {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Exercice :** sur ce même dataset déséquilibré, comparer un modèle "naïf" qui prédit toujours la classe majoritaire à un vrai modèle entraîné, en affichant les deux côte à côte sur accuracy, F1 et PR-AUC — constater que l'accuracy des deux est proche alors que le F1 et la PR-AUC révèlent l'écart réel. Puis remplacer `StratifiedKFold` par un simple `KFold` non stratifié et observer la variance plus élevée des scores entre plis.

**🎥 Vidéo :** "Quelle métrique choisir pour évaluer un modèle (et pourquoi l'accuracy ment souvent)" — démo à l'écran du modèle naïf à 99% d'accuracy inutile, puis lecture commentée d'une courbe ROC et d'une courbe précision-rappel côte à côte sur le même modèle.

---

### Semaine 9 — Clustering (1/2) : K-Means & clustering hiérarchique

**À apprendre en détail :**
- **Apprentissage non supervisé** : rupture avec tout ce qui précède — plus de `y`, l'objectif est de découvrir une structure (des groupes) dans les données à partir des seules features
- **K-Means — algorithme** : (1) initialiser `k` centroïdes, (2) assigner chaque point au centroïde le plus proche, (3) recalculer chaque centroïde comme la moyenne des points qui lui sont assignés, (4) répéter (2)-(3) jusqu'à convergence (les assignations ne changent plus)
- **Fonction objectif — l'inertie (WCSS)** : `Σ_k Σ_{x∈Cluster_k} ||x - μ_k||²` — la somme des distances au carré entre chaque point et le centroïde de son cluster. K-Means minimise cette quantité de façon gloutonne (comme l'algorithme CART de la semaine 3, il ne garantit pas l'optimum global)
- **k-means++** : stratégie d'initialisation qui choisit les centroïdes de départ en les éloignant les uns des autres (plutôt qu'un tirage purement aléatoire), ce qui accélère la convergence et réduit le risque de tomber sur un mauvais minimum local
- **Choisir k — méthode du coude (elbow method)** : tracer l'inertie en fonction de `k` ; l'inertie décroît toujours quand `k` augmente (à la limite, `k=n` donne une inertie nulle), on cherche le point où le gain marginal ralentit nettement (le "coude")
- **Limites de K-Means** : suppose des clusters de forme sphérique/convexe et de taille comparable, sensible à l'échelle des variables (standardisation indispensable, comme pour les SVM en semaine 2), sensible aux outliers (un point extrême tire fortement son centroïde), nécessite de fixer `k` à l'avance
- **Clustering hiérarchique agglomératif** : chaque point démarre comme son propre cluster ; à chaque étape, on fusionne les deux clusters les plus proches, jusqu'à n'en avoir plus qu'un seul — produit un **dendrogramme** (arbre des fusions) qu'on peut "couper" à la hauteur voulue pour obtenir n'importe quel nombre de clusters, sans avoir à ré-entraîner
- **Méthodes de linkage (distance entre deux clusters, pas entre deux points)** :
  - *Single linkage* : distance minimale entre un point de chaque cluster (peut créer des chaînes allongées)
  - *Complete linkage* : distance maximale entre un point de chaque cluster (clusters compacts)
  - *Average linkage* : moyenne de toutes les distances entre paires
  - *Ward* : fusionne les deux clusters qui minimisent l'augmentation de l'inertie intra-cluster totale — le plus utilisé en pratique, comparable en esprit à l'objectif de K-Means
- **Métriques de distance** : euclidienne (par défaut), Manhattan (moins sensible aux outliers sur une dimension), cosinus (pertinente quand seule la direction du vecteur compte, ex : texte ou comportements normalisés)

**Hyperparamètres clés (sklearn `KMeans` / `AgglomerativeClustering`) :**

| Paramètre | Rôle | Effet si mal réglé |
|---|---|---|
| `n_clusters` | Nombre de clusters cible (les deux algorithmes) | Trop bas : sous-groupes distincts fusionnés. Trop haut : sur-segmentation, clusters artificiels |
| `init` (KMeans) | Stratégie d'initialisation (`k-means++` vs `random`) | `random` : risque de mauvaise convergence, résultats instables |
| `n_init` (KMeans) | Nombre de réinitialisations, garde le meilleur résultat | Trop bas : résultat dépendant du hasard de l'initialisation |
| `linkage` (Agglomerative) | Méthode de calcul de distance inter-clusters | `single` : sensible aux effets de chaîne ; `ward` : clusters plus équilibrés |
| `distance_threshold` (Agglomerative) | Hauteur de coupe du dendrogramme (alternative à `n_clusters`) | Permet de laisser l'algorithme déterminer le nombre de clusters |

**Exemple pratique :**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 1. Méthode du coude pour choisir k
inerties = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    inerties.append(km.inertia_)

plt.figure()
plt.plot(range(1, 9), inerties, marker="o")
plt.xlabel("k"); plt.ylabel("Inertie (WCSS)")
plt.title("Méthode du coude")
plt.savefig("elbow.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. K-Means final avec k=4
kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)
print(f"Inertie finale K-Means (k=4): {kmeans.inertia_:.2f}")

# 3. Clustering hiérarchique + dendrogramme
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=15)
plt.title("Dendrogramme (linkage=ward)")
plt.savefig("dendrogramme.png", dpi=150, bbox_inches="tight")
plt.close()

agglo = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels_agglo = agglo.fit_predict(X_scaled)
print(f"Clusters trouvés (hiérarchique): {len(np.unique(labels_agglo))}")
```

**Exercice :** sur `make_blobs`, vérifier avec la méthode du coude que `k=4` correspond bien au vrai nombre de centres générés. Puis, avec `k=4` fixe, comparer les résultats de K-Means avec `linkage="single"` vs `linkage="ward"` en clustering hiérarchique sur un dataset bruité — observer les effets de chaîne du single linkage.

**🎥 Vidéo :** "K-Means et clustering hiérarchique, expliqués visuellement" — animation de la boucle assignation/recalcul de K-Means, lecture d'un dendrogramme et de sa coupe, comparaison des linkages sur un même jeu de données.

---

### Semaine 10 — Clustering (2/2) : DBSCAN, GMM & évaluation du clustering

**À apprendre en détail :**
- **DBSCAN (density-based)** : classe les points en trois catégories selon la densité locale — **point cœur** (au moins `min_samples` voisins dans un rayon `eps`), **point frontière** (dans le rayon `eps` d'un point cœur, mais pas assez de voisins pour l'être lui-même), **bruit** (aucun des deux, étiqueté `-1`) — les clusters se forment par connexité entre points cœurs
- **Avantages de DBSCAN sur K-Means** : pas besoin de fixer `k` à l'avance, détecte des clusters de forme arbitraire (non convexe, ex : `make_moons` de la semaine 2), robuste aux outliers puisqu'ils sont simplement étiquetés comme bruit plutôt que forcés dans un cluster
- **Limites de DBSCAN** : peine sur des clusters de densités très différentes (un seul couple `eps`/`min_samples` pour tout l'espace), sensible au choix de ces deux hyperparamètres
- **Gaussian Mixture Models (GMM)** : modélise les données comme un mélange de plusieurs distributions gaussiennes ; contrairement à K-Means (assignation dure), le GMM donne une **probabilité d'appartenance** à chaque cluster pour chaque point (*soft clustering*)
- **Algorithme EM (Expectation-Maximization)** — intuition : (E) étant donné les paramètres actuels des gaussiennes, calculer la probabilité que chaque point appartienne à chaque composante ; (M) étant donné ces probabilités, ré-estimer les paramètres (moyenne, covariance, poids) de chaque gaussienne pour maximiser la vraisemblance ; répéter jusqu'à convergence — même logique itérative que K-Means, mais en probabiliste
- **Types de covariance (`covariance_type`)** : `spherical` (clusters ronds, comme K-Means), `diag`, `tied`, `full` (ellipses orientées librement, la plus flexible mais la plus coûteuse)
- **Choisir le nombre de composantes** : BIC (Bayesian Information Criterion) et AIC (Akaike Information Criterion) pénalisent la vraisemblance par la complexité du modèle — on choisit le nombre de composantes qui minimise le BIC/AIC plutôt que de maximiser la seule vraisemblance (qui augmenterait indéfiniment avec plus de composantes, comme l'overfitting d'un arbre trop profond)
- **Évaluer un clustering sans étiquettes (cas le plus fréquent en pratique)** :
  - **Score de silhouette** : pour chaque point, `s = (b - a) / max(a, b)`, où `a` = distance moyenne aux points de son propre cluster, `b` = distance moyenne aux points du cluster voisin le plus proche. Varie de -1 (mal classé) à +1 (bien groupé) ; proche de 0 = clusters qui se chevauchent
  - **Indice de Davies-Bouldin** : moyenne, sur tous les clusters, du ratio entre la dispersion intra-cluster et la distance inter-cluster — plus bas est meilleur (0 = séparation parfaite)
  - **Indice de Calinski-Harabasz** : ratio entre dispersion inter-cluster et dispersion intra-cluster — plus haut est meilleur
- **Quand des vraies étiquettes existent (rare mais utile pour valider un algorithme)** : ARI (Adjusted Rand Index) et NMI (Normalized Mutual Information) comparent le clustering obtenu à un partitionnement de référence

**Hyperparamètres clés (sklearn `DBSCAN` / `GaussianMixture`) :**

| Paramètre | Rôle | Effet si mal réglé |
|---|---|---|
| `eps` (DBSCAN) | Rayon du voisinage considéré | Trop petit : presque tout devient du bruit. Trop grand : tous les clusters fusionnent en un seul |
| `min_samples` (DBSCAN) | Nombre min de voisins pour être un point cœur | Trop bas : sensible au bruit, clusters parasites. Trop haut : peu de points qualifiés de "cœur", sur-classification en bruit |
| `n_components` (GMM) | Nombre de gaussiennes du mélange | Choisi via BIC/AIC plutôt qu'au hasard |
| `covariance_type` (GMM) | Forme géométrique des clusters | `spherical` : trop rigide si clusters allongés. `full` : plus de paramètres à estimer, risque de surapprentissage sur peu de données |

**Exemple pratique :**
```python
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

X, _ = make_moons(n_samples=300, noise=0.07, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 1. K-Means échoue sur des clusters non convexes
km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_scaled)
print(f"K-Means   | silhouette: {silhouette_score(X_scaled, km.labels_):.3f}")

# 2. DBSCAN capture la forme en croissant
db = DBSCAN(eps=0.3, min_samples=5).fit(X_scaled)
n_bruit = np.sum(db.labels_ == -1)
print(f"DBSCAN    | clusters trouvés: {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)} | points de bruit: {n_bruit}")
mask = db.labels_ != -1
print(f"DBSCAN    | silhouette (hors bruit): {silhouette_score(X_scaled[mask], db.labels_[mask]):.3f}")

# 3. GMM avec sélection du nombre de composantes par BIC
bics = []
for k in range(1, 6):
    gmm = GaussianMixture(n_components=k, random_state=42).fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
best_k = np.argmin(bics) + 1
print(f"GMM | meilleur nombre de composantes selon BIC: {best_k}")
```

**Exercice :** sur `make_moons`, comparer visuellement les clusters trouvés par K-Means, DBSCAN et GMM (nuage de points coloré par label) — constater que seul DBSCAN capture correctement les deux croissants. Puis, sur `make_blobs` avec des clusters de densités différentes, montrer les limites de DBSCAN en faisant varier `eps`.

**🎥 Vidéo :** "DBSCAN et GMM : quand K-Means ne suffit plus" — démo visuelle de l'échec de K-Means sur `make_moons`, explication des points cœur/frontière/bruit de DBSCAN, puis lecture d'une courbe BIC pour choisir le nombre de composantes d'un GMM.

---

### Semaine 11 — Gérer le déséquilibre des classes

**Pourquoi cette semaine est essentielle :** fraude (souvent <1% des transactions), churn, désabonnement, défaut de paiement, détection de maladies rares — la plupart des cas d'usage métier à forte valeur sont déséquilibrés. Cette semaine s'appuie directement sur la semaine 8 (métriques) : sans les bonnes métriques, impossible de savoir si les techniques ci-dessous améliorent réellement le modèle.

**À apprendre en détail :**
- **Le problème** : un modèle entraîné naïvement sur des classes déséquilibrées apprend à privilégier la classe majoritaire car c'est ce qui minimise la loss globale — l'accuracy reste haute (semaine 8) alors que le modèle est inutilisable sur la classe qui compte
- **Approche 1 — Rééchantillonnage (resampling)** :
  - *Sous-échantillonnage aléatoire (undersampling)* : supprimer des exemples de la classe majoritaire — simple mais perd de l'information
  - *Sur-échantillonnage aléatoire (oversampling)* : dupliquer des exemples de la classe minoritaire — risque de surapprentissage sur les points dupliqués
  - **SMOTE (Synthetic Minority Oversampling Technique)** : au lieu de dupliquer, on **génère** des points synthétiques. Pour un point minoritaire `x_i`, on choisit un de ses `k` plus proches voisins minoritaires `x_zi`, et on crée un nouveau point sur le segment qui les relie : `x_new = x_i + λ · (x_zi - x_i)`, avec `λ` tiré aléatoirement dans `[0, 1]` — la démonstration géométrique du calcul se fait facilement à la main sur 2 points en 2D
  - **ADASYN** : variante de SMOTE qui génère davantage de points synthétiques dans les régions où la classe minoritaire est la plus difficile à apprendre (proche de la frontière de décision, entourée de voisins majoritaires) plutôt qu'uniformément
  - *Tomek links* : paires de points de classes opposées mutuellement plus proches l'un de l'autre que de tout autre point — les supprimer (côté majoritaire) nettoie la frontière de décision ; souvent combiné à SMOTE (`SMOTE + Tomek`)
- **Approche 2 — Pondération au niveau de l'algorithme (cost-sensitive learning)** : `class_weight="balanced"` (déjà vu en semaine 1 pour la régression logistique) pondère la fonction de perte inversement à la fréquence de chaque classe, sans dupliquer ni supprimer de données. `scale_pos_weight` en XGBoost (semaine 6) applique le même principe. Généralement à essayer **avant** le rééchantillonnage, car moins coûteux et sans risque de surapprentissage sur des points synthétiques
- **Approche 3 — Ajustement du seuil de décision (threshold tuning)** : plutôt que de modifier les données ou le modèle, on déplace le seuil de classification (par défaut 0.5) en utilisant la courbe précision-rappel de la semaine 8 pour choisir le seuil qui correspond au compromis métier voulu (ex : privilégier fortement le rappel en détection de fraude, quitte à sacrifier de la précision)
- **Évaluation adaptée** : ne jamais utiliser l'accuracy seule (semaine 8) — privilégier F1-score, PR-AUC, ou le **MCC (Matthews Correlation Coefficient)**, une métrique symétrique et fiable même sur un fort déséquilibre
- **Approches d'ensemble spécialisées** : `BalancedRandomForestClassifier` (sous-échantillonne chaque bootstrap avant d'entraîner chaque arbre de la semaine 4), `EasyEnsemble` (combine sous-échantillonnage et boosting)

**Hyperparamètres clés (`imbalanced-learn`, sklearn) :**

| Paramètre | Rôle | Effet si mal réglé |
|---|---|---|
| `sampling_strategy` | Ratio cible entre classe minoritaire et majoritaire après rééchantillonnage | À 1.0 (parfait équilibre) : risque de sur-corriger et de créer trop de points synthétiques irréalistes |
| `k_neighbors` (SMOTE) | Nombre de voisins utilisés pour l'interpolation | Trop bas : points synthétiques peu variés. Trop haut : interpolation entre points trop éloignés, points synthétiques peu réalistes |
| `class_weight` | Pondération inverse de la fréquence de classe | `None` : la classe rare est ignorée par la loss |

**Exemple pratique :**
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# Dataset très déséquilibré (1% de positifs, ex: fraude)
X, y = make_classification(
    n_samples=5000, n_features=10, weights=[0.99, 0.01], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Répartition originale train: {np.bincount(y_train)}")

# 1. Baseline sans traitement du déséquilibre
baseline = RandomForestClassifier(random_state=42).fit(X_train, y_train)
proba_base = baseline.predict_proba(X_test)[:, 1]
print(f"Baseline        | F1: {f1_score(y_test, baseline.predict(X_test)):.3f} | PR-AUC: {average_precision_score(y_test, proba_base):.3f}")

# 2. class_weight="balanced" (à essayer en premier)
weighted = RandomForestClassifier(class_weight="balanced", random_state=42).fit(X_train, y_train)
proba_w = weighted.predict_proba(X_test)[:, 1]
print(f"class_weight    | F1: {f1_score(y_test, weighted.predict(X_test)):.3f} | PR-AUC: {average_precision_score(y_test, proba_w):.3f}")

# 3. SMOTE
X_res, y_res = SMOTE(k_neighbors=5, random_state=42).fit_resample(X_train, y_train)
print(f"Répartition après SMOTE: {np.bincount(y_res)}")
smote_model = RandomForestClassifier(random_state=42).fit(X_res, y_res)
proba_smote = smote_model.predict_proba(X_test)[:, 1]
print(f"SMOTE           | F1: {f1_score(y_test, smote_model.predict(X_test)):.3f} | PR-AUC: {average_precision_score(y_test, proba_smote):.3f}")

# 4. Ajustement du seuil sur le modèle baseline
precisions, recalls, thresholds = precision_recall_curve(y_test, proba_base)
# Choisir le seuil qui garantit un rappel minimum de 0.8
idx = np.argmax(recalls[:-1] >= 0.8)
print(f"Seuil ajusté pour rappel>=0.8: {thresholds[idx]:.3f} (au lieu de 0.5)")
```

**Exercice :** sur ce dataset à 1% de positifs, comparer les 4 approches (baseline, `class_weight`, SMOTE, ajustement de seuil) uniquement sur F1 et PR-AUC — jamais sur l'accuracy. Identifier laquelle apporte le plus de gain sur ce cas précis, et formuler une hypothèse sur pourquoi (ex : SMOTE aide surtout quand la frontière de décision est complexe, `class_weight` suffit souvent pour des modèles linéaires).

**🎥 Vidéo :** "Comment gérer un dataset déséquilibré (fraude, churn, désabonnement)" — démo de l'échec silencieux de l'accuracy sur données déséquilibrées, animation du mécanisme d'interpolation de SMOTE, puis comparatif des 4 approches sur un même jeu de données avec PR-AUC.

---

## PHASE 2 — Deep Learning & PyTorch (Semaines 12-18)

### Semaine 12 — Bases de PyTorch
**À apprendre en détail :**
- Tensors : création, opérations, broadcasting, passage CPU ↔ GPU
- Autograd : graphe de calcul dynamique, `requires_grad`, `.backward()`
- `Dataset` et `DataLoader` : pipeline de données, batching, shuffling
- `nn.Module` : structure d'un modèle personnalisé

**Exercice :** implémenter une régression linéaire "from scratch" avec autograd (sans `nn.Linear`) pour comprendre le calcul du gradient.

**🎥 Vidéo :** "PyTorch de zéro : tensors et autograd expliqués" — notebook commenté.

---

### Semaine 13 — Entraîner un réseau de neurones (MLP)
**À apprendre en détail :**
- Boucle d'entraînement complète : forward → loss → backward → step
- Fonctions de perte : `CrossEntropyLoss`, `MSELoss` et leur usage
- Optimiseurs : SGD, Momentum, RMSprop, **Adam** (comprendre les moments m1/m2)
- Techniques anti-overfitting : Dropout, weight decay (L2), early stopping
- Suivi d'entraînement avec TensorBoard

**Exercice :** entraîner un MLP sur MNIST, suivre les courbes de loss/accuracy avec TensorBoard.

**🎥 Vidéo :** "Anatomie d'une boucle d'entraînement PyTorch" — code à l'écran, ligne par ligne.

---

### Semaine 14 — CNN (réseaux convolutionnels)
**À apprendre en détail :**
- Convolution : filtres, stride, padding, feature maps
- Pooling (max/average), Flatten
- Architectures de référence : VGG, **ResNet** (connexions résiduelles — pourquoi elles règlent le *vanishing gradient*), EfficientNet
- Transfer learning : fine-tuning d'un modèle pré-entraîné (`torchvision.models`)

**Exercice :** classification d'images par transfer learning (ResNet pré-entraîné + fine-tuning sur un petit dataset custom).

**🎥 Vidéo :** "Transfer Learning en 15 minutes avec PyTorch" — démo notebook.

---

### Semaine 15 — RNN, LSTM, GRU
**À apprendre en détail :**
- Principe des réseaux récurrents et le problème du *vanishing/exploding gradient*
- LSTM : portes d'oubli, d'entrée, de sortie — comment elles résolvent le problème
- GRU : version simplifiée du LSTM
- Cas d'usage : séries temporelles, texte (avant les Transformers)

**Exercice :** prédiction de série temporelle avec un LSTM (ex : consommation électrique).

**🎥 Vidéo :** "LSTM expliqué avec un schéma animé" — pas de code cette semaine, focus pédagogique visuel.

---

### Semaine 16 — Le mécanisme d'attention & Transformer
**À apprendre en détail (semaine charnière — à ne pas bâcler)**
- Self-attention : Query, Key, Value — calcul du score d'attention (produit scalaire mis à l'échelle)
- Multi-head attention : pourquoi plusieurs têtes captent différents types de relations
- Positional encoding : pourquoi nécessaire (le Transformer n'a pas de notion d'ordre native)
- Architecture complète : blocs encoder/decoder, connexions résiduelles, layer norm

**Exercice :** implémenter un bloc de self-attention "from scratch" en PyTorch (sans librairie) sur une petite séquence.

**🎥 Vidéo :** "Le mécanisme d'attention, enfin compris" — animation schématique du calcul Q/K/V.

---

### Semaine 17 — Les familles de Transformers & Hugging Face
**À apprendre en détail :**
- Encoder-only (BERT) : compréhension bidirectionnelle, cas d'usage classification/NER
- Decoder-only (GPT) : génération autorégressive
- Encoder-decoder (T5, BART) : traduction, résumé
- Tokenisation : BPE, WordPiece, SentencePiece
- Librairie `transformers` (Hugging Face) : `AutoModel`, `AutoTokenizer`, `Trainer` API

**Exercice :** fine-tuner un BERT pré-entraîné pour une tâche de classification de texte (Hugging Face `Trainer`).

**🎥 Vidéo :** "Fine-tuner un BERT en 20 lignes avec Hugging Face" — démo pratique.

---

### Semaine 18 — Projet de consolidation Deep Learning
**À apprendre en détail :**
- Structurer un projet DL de bout en bout : préparation données → entraînement → évaluation → sauvegarde du modèle (`torch.save`)
- Bonnes pratiques : reproductibilité (seed), checkpointing, gestion GPU/CPU

**Exercice :** projet complet au choix (classification d'images ou de texte) présenté comme un mini-portfolio.

**🎥 Vidéo :** "Mon premier projet Deep Learning de bout en bout" — bilan + démo du modèle final.

---

## PHASE 3 — LLM, RAG & Agents IA (Semaines 19-26)

### Semaine 19 — Fondamentaux des LLM
**À apprendre en détail :**
- Pré-entraînement (next-token prediction) vs fine-tuning vs alignement
- Scaling laws (relation taille modèle / données / performance)
- Fenêtre de contexte, tokens, coût associé
- Paramètres de génération : température, top-p, top-k, repetition penalty

**🎥 Vidéo :** "Comment un LLM génère du texte, mot par mot" — explication du sampling avec exemples.

---

### Semaine 20 — Fine-tuning efficace : LoRA & QLoRA
**À apprendre en détail :**
- Pourquoi le fine-tuning complet est coûteux (mise à jour de milliards de paramètres)
- **LoRA** : décomposition en matrices de faible rang, injection dans les couches d'attention
- **QLoRA** : quantization 4-bit du modèle de base + LoRA par-dessus
- Librairie `peft` (Hugging Face)

**Exercice :** fine-tuner un petit LLM open-source (ex : Llama 3 8B ou Mistral 7B) avec QLoRA sur Google Colab.

**🎥 Vidéo :** "Fine-tuner un LLM avec 1 seul GPU grâce à QLoRA" — démo Colab commentée.

---

### Semaine 21 — Prompt Engineering
**À apprendre en détail :**
- Zero-shot vs few-shot prompting
- Chain-of-Thought (CoT) : pourquoi ça améliore le raisonnement
- Structured output (JSON mode), contraintes de format
- Système de prompts (system/user/assistant), prompt templates réutilisables

**Exercice :** construire une bibliothèque de prompts réutilisables pour une tâche métier (ex : extraction d'information structurée).

**🎥 Vidéo :** "5 techniques de prompt engineering qui changent tout" — exemples avant/après.

---

### Semaine 22 — Embeddings & recherche vectorielle
**À apprendre en détail :**
- Modèles d'embeddings (sentence-transformers, embeddings OpenAI)
- Similarité cosinus, distance euclidienne
- Algorithmes ANN (*Approximate Nearest Neighbor*) : HNSW
- Bases vectorielles : FAISS (local), Chroma, comparaison avec Pinecone/Weaviate

**Exercice :** indexer un corpus de documents dans FAISS et effectuer une recherche sémantique.

**🎥 Vidéo :** "Comment un moteur de recherche sémantique fonctionne" — schéma + démo FAISS.

---

### Semaine 23 — RAG (Retrieval-Augmented Generation)
**À apprendre en détail :**
- Stratégies de *chunking* (taille fixe, sémantique, par section)
- Pipeline complet : ingestion → embedding → indexation → retrieval → génération
- *Re-ranking* et recherche hybride (BM25 + vecteurs)
- Évaluation d'un RAG : fidélité (*faithfulness*), pertinence du contexte

**Exercice :** construire un RAG complet sur un corpus de documents personnels (PDF) avec LangChain + FAISS.

**🎥 Vidéo :** "Je construis un RAG de A à Z" — démo pipeline complet.

---

### Semaine 24 — Fondamentaux des Agents IA
**À apprendre en détail :**
- Pattern **ReAct** (Reason + Act) : alternance raisonnement/action
- Function calling / tool use : comment un LLM déclenche un outil externe
- Mémoire d'agent : court terme (contexte) vs long terme (base vectorielle)
- Boucle de planification (*Plan-and-Execute*)

**Exercice :** créer un agent simple avec un outil (ex : calculatrice, recherche web) en appel de fonction natif.

**🎥 Vidéo :** "Comment un agent IA 'décide' d'utiliser un outil" — trace d'exécution commentée.

---

### Semaine 25 — Frameworks d'agents & MCP
**À apprendre en détail :**
- Comparatif LangChain / **LangGraph** (orchestration en graphe d'états) / CrewAI (agents multi-rôles)
- **MCP (Model Context Protocol)** : architecture client-serveur standardisée pour exposer des outils/ressources à un LLM, différence avec le function calling propriétaire
- Construction d'un serveur MCP simple

**Exercice :** construire un agent multi-étapes avec LangGraph, et un serveur MCP basique exposant un outil.

**🎥 Vidéo :** "MCP expliqué : le 'USB-C' des agents IA" — schéma d'architecture.

---

### Semaine 26 — Projet capstone LLM
**À apprendre en détail :**
- Assemblage complet : RAG + agent + outils, interface utilisateur simple (Streamlit/Gradio)

**Exercice :** projet complet (ex : assistant documentaire d'entreprise avec agent + RAG), déployé en démo locale.

**🎥 Vidéo :** "Mon assistant IA de bout en bout : RAG + Agent" — démo du produit fini.

---

## PHASE 4 — MLOps (Semaines 27-31)

### Semaine 27 — MLflow
**À apprendre en détail :**
- Tracking d'expériences : paramètres, métriques, artefacts
- Model Registry : versioning, staging → production
- Comparaison d'expériences via l'UI MLflow

**Exercice :** tracker tous les entraînements de boosting (Phase 1) rétroactivement dans MLflow.

**🎥 Vidéo :** "Suivre ses expériences ML comme un pro avec MLflow"

---

### Semaine 28 — Airflow
**À apprendre en détail :**
- Concept de DAG (*Directed Acyclic Graph*), opérateurs, tâches
- Scheduling et dépendances entre tâches
- Orchestration d'un pipeline data → entraînement → évaluation

**Exercice :** créer un DAG Airflow qui automatise le pipeline d'entraînement d'un modèle.

**🎥 Vidéo :** "Automatiser un pipeline ML avec Airflow" — démo du DAG en exécution.

---

### Semaine 29 — Docker
**À apprendre en détail :**
- Concepts : image, conteneur, `Dockerfile`, layers
- Bonnes pratiques : multi-stage build, taille d'image, `.dockerignore`
- Conteneuriser une application Python/ML

**Exercice :** dockeriser le projet RAG de la semaine 23.

**🎥 Vidéo :** "Dockeriser une application IA en 10 minutes"

---

### Semaine 30 — FastAPI
**À apprendre en détail :**
- Création d'endpoints REST, validation avec Pydantic
- Endpoints asynchrones (`async def`), gestion de la latence pour l'inférence
- Documentation automatique (Swagger/OpenAPI)

**Exercice :** exposer le modèle XGBoost (Phase 1) et le RAG via une API FastAPI, conteneurisée avec Docker.

**🎥 Vidéo :** "Déployer un modèle ML en API avec FastAPI + Docker"

---

### Semaine 31 — CI/CD & monitoring
**À apprendre en détail :**
- Pipeline CI/CD avec GitHub Actions (tests automatiques, build Docker, déploiement)
- Notions de *data drift* / *model drift*
- Introduction à Evidently AI pour le monitoring de modèles en production

**Exercice :** mettre en place une pipeline GitHub Actions qui build et teste l'API FastAPI automatiquement.

**🎥 Vidéo :** "CI/CD pour un projet ML : mon pipeline GitHub Actions"

---

## PHASE 5 — Cloud & Kubernetes (Semaines 32-34)

### Semaine 32 — Cloud ML (Azure / AWS)
**À apprendre en détail :**
- Azure ML : workspace, compute, endpoints managés
- AWS SageMaker : training jobs, endpoints d'inférence
- AWS Bedrock / Azure OpenAI : accès managé aux LLM

**Exercice :** déployer le modèle boosting comme endpoint managé sur Azure ML ou SageMaker.

**🎥 Vidéo :** "Déployer un modèle sur Azure ML / SageMaker, pas à pas"

---

### Semaine 33 — Kubernetes
**À apprendre en détail :**
- Concepts : Pod, Deployment, Service, ConfigMap, Ingress
- Scaling horizontal, self-healing
- Déploiement local avec Minikube

**Exercice :** déployer l'API FastAPI conteneurisée (semaine 30) sur un cluster Minikube local.

**🎥 Vidéo :** "Kubernetes pour Data Scientists : les bases essentielles"

---

### Semaine 34 — Projet de synthèse final
**À apprendre en détail :**
- Assembler la chaîne complète : entraînement → tracking (MLflow) → orchestration (Airflow) → conteneurisation (Docker) → API (FastAPI) → déploiement (Kubernetes/Cloud) → monitoring

**Exercice :** projet final end-to-end présentant l'ensemble de la chaîne MLOps sur un cas d'usage LLM ou ML classique.

**🎥 Vidéo :** "Bilan de 6 mois : ma chaîne MLOps complète, de A à Z" — vidéo de synthèse/portfolio.

---

## Résumé de la cadence

| Phase | Semaines | Thème |
|---|---|---|
| 1 | 1-11 | Fondamentaux classification (régression logistique, SVM, arbre) + Ensembles (Bagging/Random Forest, Boosting/XGBoost/LightGBM/CatBoost) + SHAP/LIME + Métriques d'évaluation + Clustering (K-Means, hiérarchique, DBSCAN, GMM) + Déséquilibre des classes |
| 2 | 12-18 | Deep Learning + PyTorch |
| 3 | 19-26 | LLM, RAG, Agents IA, MCP |
| 4 | 27-31 | MLOps (MLflow, Airflow, Docker, FastAPI, CI/CD) |
| 5 | 32-34 | Cloud & Kubernetes |

**Conseil pour les vidéos :** gardez un format court (8-12 min), toujours structuré en 3 temps — "le concept" (schéma/animation) → "la démo" (code/terminal à l'écran) → "ce qu'il faut retenir". Cela crée une série cohérente et facile à suivre sur les 29 semaines.
