# 🌹 Semantic RAG | Rose Blanche Group

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-enabled-00C853?style=for-the-badge)

**Module Intelligent de Recherche Sémantique (RAG)**

*Challenge RoboKids 84 Explorers — Prix: 1000 DT*

[Démarrage Rapide](#-démarrage-rapide) •
[Fonctionnalités](#-fonctionnalités) •
[Architecture](#-architecture) •
[Documentation](#-documentation)

---

<img src="https://img.icons8.com/color/200/000000/search-in-cloud.png" alt="Semantic Search" width="150"/>

</div>

---

## 🎯 Contexte

Dans le cadre d'un projet d'**assistance intelligente à la formulation en boulangerie et pâtisserie**, ce module permet de retrouver automatiquement les fragments les plus pertinents d'une base documentaire à partir d'une question formulée en **langage naturel**.

### ❌ Problème
> Les utilisateurs rencontrent des difficultés à identifier rapidement les passages pertinents dans un grand volume d'informations techniques.

### ✅ Solution
> Un module **RAG** (Retrieval-Augmented Generation) qui privilégie la **proximité sémantique** plutôt qu'une simple correspondance lexicale.

---

## ✨ Fonctionnalités

<table>
<tr>
<td width="50%">

### 🔤 Traitement du Langage
- Réception de questions en langage naturel
- Génération d'embeddings sémantiques (384 dimensions)
- Support français et anglais

</td>
<td width="50%">

### 🔍 Recherche Intelligente
- Similarité cosinus pour le matching
- Classement par pertinence décroissante
- Déduplication automatique des résultats

</td>
</tr>
<tr>
<td width="50%">

### 💾 Base de Données
- PostgreSQL avec extension pgvector
- Stockage vectoriel optimisé
- Cache local pour performance

</td>
<td width="50%">

### 🖥️ Interface
- CLI interactive
- **Web UI premium** (Streamlit)
- Affichage texte + score

</td>
</tr>
</table>

---

## 🏗 Architecture

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   👤 UTILISATEUR                                                          ║
║   "Quel est le dosage recommandé d'alpha-amylase ?"                      ║
║                           │                                               ║
║                           ▼                                               ║
║   ┌─────────────────────────────────────────────────────────────────┐    ║
║   │                    MODULE RAG                                    │    ║
║   ├─────────────────────────────────────────────────────────────────┤    ║
║   │                                                                  │    ║
║   │   ┌──────────┐    ┌──────────────┐    ┌──────────────┐         │    ║
║   │   │  ÉTAPE 1 │───▶│   ÉTAPE 2    │───▶│   ÉTAPE 3    │         │    ║
║   │   │ Question │    │  Embedding   │    │   Cosine     │         │    ║
║   │   └──────────┘    └──────────────┘    │  Similarity  │         │    ║
║   │                          │            └──────────────┘         │    ║
║   │                          ▼                   │                  │    ║
║   │                 ┌──────────────┐             ▼                  │    ║
║   │                 │ all-MiniLM   │    ┌──────────────┐           │    ║
║   │                 │   -L6-v2     │    │   ÉTAPE 4    │           │    ║
║   │                 │  (384 dim)   │    │   Ranking    │           │    ║
║   │                 └──────────────┘    └──────────────┘           │    ║
║   │                                            │                    │    ║
║   │                                            ▼                    │    ║
║   │                                   ┌──────────────┐             │    ║
║   │                                   │  ÉTAPE 5-6   │             │    ║
║   │                                   │  Top 3 +     │             │    ║
║   │                                   │  Affichage   │             │    ║
║   │                                   └──────────────┘             │    ║
║   │                                                                  │    ║
║   └─────────────────────────────────────────────────────────────────┘    ║
║                           │                                               ║
║                           ▼                                               ║
║   ┌─────────────────────────────────────────────────────────────────┐    ║
║   │                 PostgreSQL + pgvector                            │    ║
║   │  ┌─────────────────────────────────────────────────────────┐    │    ║
║   │  │ Table: embeddings                                        │    │    ║
║   │  │ ├── id (Primary Key)                                     │    │    ║
║   │  │ ├── id_document (int)                                    │    │    ║
║   │  │ ├── texte_fragment (text)                                │    │    ║
║   │  │ └── vecteur (VECTOR(384))                                │    │    ║
║   │  └─────────────────────────────────────────────────────────┘    │    ║
║   └─────────────────────────────────────────────────────────────────┘    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 🚀 Démarrage Rapide

### Prérequis

| Outil | Version | Description |
|-------|---------|-------------|
| Python | 3.8+ | Langage de programmation |
| Docker | Latest | Pour PostgreSQL |
| pip | Latest | Gestionnaire de paquets |

### Installation

```bash
# 1. Cloner le projet
git clone https://github.com/YassineMabrou/semantic_rag1.git
cd semantic_rag1

# 2. Créer l'environnement virtuel
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer PostgreSQL avec pgvector
docker run -d \
  --name my-postgres \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=database \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 5. Initialiser les données
python insert_embeddings.py
```

### Lancement

<table>
<tr>
<td width="50%">

#### 🖥️ Interface Web (Recommandé)
```bash
streamlit run app.py
```
Ouvrir: `http://localhost:8501`

</td>
<td width="50%">

#### ⌨️ Interface CLI
```bash
python rag_module.py
```

</td>
</tr>
</table>

---

## 📁 Structure du Projet

```
semantic_rag/
│
├── 🎨 INTERFACE
│   └── app.py                  # Interface Web Streamlit (Premium UI)
│
├── 🧠 CORE MODULE
│   ├── rag_module.py           # Module principal (PostgreSQL)
│   ├── simple_search.py        # Version simplifiée
│   └── search_local.py         # Recherche avec index local
│
├── 🔧 UTILITIES
│   ├── build_index.py          # Construction d'index depuis PDFs
│   ├── insert_embeddings.py    # Insertion en base de données
│   ├── check_db.py             # Vérification connexion DB
│   └── test_local.py           # Tests sans base de données
│
├── ⚙️ CONFIGURATION
│   ├── config.py               # Configuration centralisée
│   ├── requirements.txt        # Dépendances Python
│   └── .env.example            # Template variables d'environnement
│
├── 📦 DATA
│   ├── vector_index.pkl        # Index vectoriel local (généré)
│   └── data/enzymes/           # Fiches techniques PDF (35 fichiers)
│
└── 📚 DOCUMENTATION
    └── README.md               # Ce fichier
```

---

## 🔧 Spécifications Techniques

### Modèle d'Embedding

| Paramètre | Valeur |
|-----------|--------|
| **Modèle** | `all-MiniLM-L6-v2` |
| **Bibliothèque** | `sentence-transformers` |
| **Dimension** | 384 |
| **Normalisation** | ✅ Activée |

### Paramètres de Recherche

| Paramètre | Valeur |
|-----------|--------|
| **Méthode** | Similarité Cosinus |
| **Résultats** | Top K = 3 |
| **Déduplication** | ✅ Activée |

### Formule de Similarité

```
                    A · B
cos(θ) = ─────────────────────────
              ‖A‖ × ‖B‖

Pour vecteurs normalisés: cos(θ) = A · B
```

---

## 📝 Exemples d'Utilisation

### Question 1: Dosage d'enzymes

**Question:**
```
Améliorant de panification : quelles sont les quantités recommandées 
d'alpha-amylase, xylanase et d'Acide ascorbique ?
```

**Résultat:**
```
Résultat 1
Texte : "Dosage recommandé : 0.005% à 0.02% du poids de farine."
Score : 0.91

Résultat 2
Texte : "Alpha-amylase : utilisation entre 5 et 20 ppm selon la farine."
Score : 0.87

Résultat 3
Texte : "Xylanase : améliore l'extensibilité de la pâte…"
Score : 0.82
```

### Question 2: Amélioration de la pâte

**Question:**
```
Comment améliorer l'extensibilité de la pâte à pain ?
```

**Résultat:**
```
Résultat 1
Texte : "La xylanase améliore l'extensibilité de la pâte et le volume du pain"
Score : 0.85

Résultat 2
Texte : "Les transglutaminases améliorent la texture et la structure de la mie"
Score : 0.71

Résultat 3
Texte : "L'acide ascorbique agit comme agent oxydant pour renforcer le réseau gluten"
Score : 0.68
```

---

## 🎨 Interface Web

L'interface Streamlit offre une expérience utilisateur premium:

<table>
<tr>
<td align="center">
<strong>🌙 Design Dark Luxury</strong><br>
Thème sombre avec accents dorés
</td>
<td align="center">
<strong>⚡ Recherche Instantanée</strong><br>
Résultats en temps réel
</td>
<td align="center">
<strong>📊 Métriques Visuelles</strong><br>
Scores avec barres de progression
</td>
</tr>
</table>

### Fonctionnalités UI

- 🔍 Barre de recherche intuitive
- 💡 Questions suggérées (boutons rapides)
- 📈 Affichage des scores avec indicateurs visuels
- 📊 Statistiques en temps réel dans la sidebar
- 🎯 Animations fluides sur les résultats

---

## 📚 API Reference

### `generate_query_embedding(question: str) -> np.ndarray`
Génère l'embedding vectoriel d'une question.

```python
embedding = generate_query_embedding("Quel est le dosage d'alpha-amylase ?")
# Returns: numpy array shape (384,)
```

### `cosine_similarity(vec_a, vec_b) -> float`
Calcule la similarité cosinus entre deux vecteurs.

```python
score = cosine_similarity(query_vec, doc_vec)
# Returns: float [-1, 1]
```

### `semantic_search(question, fragments, vectors, top_k) -> list`
Effectue une recherche sémantique complète.

```python
results = semantic_search("Dosage alpha-amylase ?", fragments, vectors, top_k=3)
# Returns: [{"texte": "...", "score": 0.89}, ...]
```

---

## 🧪 Tests

```bash
# Vérifier la connexion DB
python check_db.py

# Test avec données locales
python test_local.py

# Test avec PDFs
python build_index.py
python search_local.py
```

---

## 🔧 Dépannage

<details>
<summary><strong>❌ "No module named 'sentence_transformers'"</strong></summary>

```bash
pip install sentence-transformers
```
</details>

<details>
<summary><strong>❌ "connection refused" (PostgreSQL)</strong></summary>

```bash
# Vérifier Docker
docker ps

# Relancer le conteneur
docker start my-postgres
```
</details>

<details>
<summary><strong>❌ "password authentication failed"</strong></summary>

```bash
# Vérifier les credentials
docker inspect my-postgres | findstr POSTGRES
```
</details>

<details>
<summary><strong>❌ "'streamlit' is not recognized"</strong></summary>

```bash
pip install streamlit
# ou
python -m streamlit run app.py
```
</details>

---

## 📦 Dépendances

```txt
sentence-transformers==2.2.2    # Modèle d'embedding
numpy>=1.24.0                   # Opérations vectorielles
psycopg2-binary>=2.9.9          # Connexion PostgreSQL
PyPDF2>=3.0.0                   # Extraction PDF
streamlit>=1.28.0               # Interface Web
```

---

## 🏆 Challenge RoboKids

<div align="center">

| Information | Détail |
|-------------|--------|
| **Challenge** | RoboKids 84 Explorers |
| **Organisation** | STE AGRO MELANGE TECHNOLOGIE |
| **Groupe** | Rose Blanche Group |
| **Contact** | a.changuel@rose-blanche.com |
| **Prix** | 🏆 Chèque cadeau de **1000 DT** |
| **Livrable** | Prototype fonctionnel |

</div>

---

## ✅ Conformité Challenge

| Exigence | Implémentation | Status |
|----------|----------------|--------|
| Recevoir question utilisateur | `input()` / Streamlit UI | ✅ |
| Générer embedding sémantique | `all-MiniLM-L6-v2` | ✅ |
| Calculer similarité cosinus | `cosine_similarity()` | ✅ |
| Classer par score décroissant | `np.argsort()[::-1]` | ✅ |
| Retourner 3 fragments | `TOP_K = 3` | ✅ |
| Afficher texte + score | Format exact respecté | ✅ |
| Base PostgreSQL + pgvector | Docker container | ✅ |
| Interface utilisateur | Web UI Premium | ✅ **BONUS** |

---

## 👤 Auteur

**Yassine Mabrou**

- GitHub: [@YassineMabrou](https://github.com/YassineMabrou)

---

<div align="center">

---

**Développé avec ❤️ pour le Challenge RoboKids 2024**

🌹 *Rose Blanche Group*

---

[⬆ Retour en haut](#-semantic-rag--rose-blanche-group)

</div>
