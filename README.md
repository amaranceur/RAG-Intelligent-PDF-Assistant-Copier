# Intelligent PDF Question Answering System Using RAG

**Système intelligent de questions-réponses sur des documents PDF utilisant la Retrieval-Augmented Generation (RAG).**

L'application permet à un utilisateur d'importer un ou plusieurs fichiers PDF, de poser des questions en langage naturel et d'obtenir des réponses fondées sur le contenu des documents, accompagnées des sources utilisées (nom du fichier et numéro de page).

---

## 1. Présentation du projet

Les modèles de langage (LLM) peuvent générer des réponses incorrectes ou inventer des informations lorsqu'ils ne disposent pas du contexte nécessaire.

Pour limiter ce problème, ce projet utilise une architecture **RAG (Retrieval-Augmented Generation)**.

Au lieu de demander directement au LLM de répondre, le système :

- Extrait le contenu des documents PDF.
- Découpe le texte en petits passages (*chunks*).
- Transforme les passages en vecteurs numériques (*embeddings*).
- Recherche les passages les plus pertinents avec FAISS.
- Transmet le contexte récupéré au LLM via l'API Groq.
- Génère une réponse accompagnée de ses sources.

Si l'information n'est pas présente dans les documents, le système doit répondre :

> Je ne trouve pas cette information dans les documents fournis.

## 2. Architecture du système RAG

Le projet implémente les trois étapes fondamentales de la RAG :

```text
              DOCUMENTS PDF
                    |
                    v
              PyMuPDF
          Extraction du texte
                    |
                    v
              Prétraitement
                    |
                    v
                Chunking
          (600 caractères,
           overlap de 100)
                    |
                    v
          Sentence Transformers
               Embeddings
                    |
                    v
                FAISS
             Indexation
                    |
                    |
QUESTION ------------+
    |
    v
Embedding de la question
    |
    v
RETRIEVAL (R)
Recherche des passages pertinents
    |
    v
AUGMENTATION (A)
Ajout du contexte au prompt
    |
    v
GENERATION (G)
       Groq API
          |
          v
   Réponse générée
          |
          v
    Sources utilisées
```

### Les trois composants RAG

| Composant | Fonction |
|---|---|
| Retrieval | Recherche des passages pertinents dans les documents. |
| Augmentation | Ajout des passages récupérés au contexte envoyé au LLM. |
| Generation | Génération d'une réponse à partir du contexte fourni. |

**Important :** Groq est utilisé uniquement comme service de génération du texte. Le système de recherche documentaire et le pipeline RAG sont développés dans le projet.

## 3. Technologies utilisées

| Composant | Technologie |
|---|---|
| Langage | Python |
| Extraction PDF | PyMuPDF |
| Prétraitement | Python |
| Embeddings | Sentence Transformers |
| Modèle d'embedding actuel | all-MiniLM-L6-v2 |
| Recherche vectorielle | FAISS (faiss-cpu) |
| Modèle de langage | Groq API |
| Interface utilisateur | Gradio |
| Environnement | Google Colab |

### Modèles de génération

Le projet utilise une liste de modèles Groq :

- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`
- `qwen/qwen3.6-27b`

Le notebook vérifie les modèles disponibles sur le compte et sélectionne un modèle candidat. Un mécanisme de repli permet d'essayer un autre modèle en cas de refus ou d'indisponibilité.

La liste peut être modifiée dans la cellule 3 du notebook.

## 4. Fonctionnement détaillé

### 4.1 Extraction des documents

La bibliothèque PyMuPDF permet de :

- Lire les fichiers PDF.
- Extraire le texte de chaque page.
- Conserver le nom du fichier et le numéro de page.
- Détecter les fichiers invalides ou sans texte extractible.

### 4.2 Prétraitement et Chunking

Le texte extrait est nettoyé puis découpé en petits passages.

Paramètres utilisés :

| Paramètre | Valeur |
|---|---:|
| Taille maximale d'un chunk | 600 caractères |
| Overlap | 100 caractères |
| Découpage | Page par page |

Le chevauchement permet de conserver une partie du contexte entre deux passages successifs.

### 4.3 Génération des embeddings

Sentence Transformers transforme chaque passage en un vecteur numérique de 384 dimensions.

Ces vecteurs représentent le contenu sémantique des textes et permettent de comparer les passages avec une question.

### 4.4 Indexation avec FAISS

FAISS permet de stocker les embeddings et de rechercher rapidement les passages les plus proches d'une question.

Le projet utilise :

- `IndexFlatIP`
- Des vecteurs normalisés.
- Le produit scalaire, équivalent à la similarité cosinus lorsque les vecteurs sont normalisés.

### 4.5 Retrieval

La question de l'utilisateur est transformée en embedding.

FAISS recherche ensuite les 4 passages les plus proches (`TOP_K = 4`).

### 4.6 Augmentation du contexte

Les passages récupérés sont ajoutés au prompt envoyé au modèle de langage.

Le modèle reçoit donc les informations extraites des PDF avant de générer sa réponse.

### 4.7 Génération de la réponse

Groq API reçoit le prompt et génère une réponse.

Le système impose plusieurs règles :

- Utiliser uniquement les informations du contexte.
- Ne pas inventer de réponses.
- Signaler l'absence d'information.
- Identifier les sources réellement utilisées.

Les sources affichées comprennent le nom du PDF et le numéro de page.

### 4.8 Gestion des erreurs

Le système prévoit plusieurs situations :

- Clé API absente ou invalide.
- Limite de requêtes atteinte (429).
- Modèle indisponible.
- Délai d'attente dépassé.
- PDF corrompu ou protégé.
- PDF sans texte extractible.

Un seuil de similarité (`MIN_SCORE = 0.15`) permet également de rejeter les questions lorsque les passages récupérés sont trop éloignés de la question.

Ce seuil reste à calibrer sur un jeu de documents représentatif.

## 5. Installation et configuration

### Google Colab

1. Ouvrir `Intelligent_PDF_QA_RAG.ipynb` dans Google Colab.
2. Installer les bibliothèques nécessaires en exécutant les cellules.
3. Créer une clé API sur : https://console.groq.com/keys
4. Dans Colab, ouvrir la section **Secrets**.
5. Ajouter un secret nommé `GROQ_API_KEY`.
6. Coller la clé API et autoriser son utilisation dans le notebook.
7. Exécuter toutes les cellules.

La clé API est récupérée depuis les Secrets de Colab et n'est pas écrite directement dans le notebook.

### Installation locale (facultative)

```bash
pip install -r requirements.txt
```

Définir ensuite la variable d'environnement :

```bash
export GROQ_API_KEY="votre_clé"
```

Puis lancer le notebook avec Jupyter.

## 6. Interface utilisateur

L'application utilise **Gradio** pour proposer une interface simple.

Les principales fonctionnalités sont :

1. Importer un ou plusieurs fichiers PDF.
2. Indexer les documents.
3. Poser des questions en langage naturel.
4. Obtenir une réponse générée à partir des documents.
5. Consulter les sources utilisées.
6. Réinitialiser l'index des documents.

## 7. Documents de démonstration

Deux documents PDF fictifs sont fournis pour tester le système.

| Document | Contenu |
|---|---|
| `demo_atlas_robotics.pdf` | Présentation d'une entreprise fictive spécialisée dans la robotique agricole. |
| `demo_politique_cybersecurite.pdf` | Politique de sécurité informatique d'une entreprise fictive. |

Chaque document contient 3 pages et des informations inventées, ce qui permet de vérifier les réponses attendues.

## 8. Tests et évaluation

Le notebook contient des tests unitaires et des tests fonctionnels portant sur les principales étapes du système.

### Tests réalisés

- Extraction du texte PDF.
- Découpage en chunks.
- Génération des embeddings.
- Indexation et recherche FAISS.
- Question avec réponse présente.
- Question avec information absente.
- Gestion de plusieurs PDF.
- Gestion des erreurs.

### Résultats de l'environnement de développement

Les tests ont été exécutés dans un environnement sans accès à Hugging Face ni à l'API Groq. Certains composants ont donc été remplacés pour tester la logique du programme.

| Élément | État |
|---|---|
| PyMuPDF | Exécuté |
| FAISS | Exécuté |
| Pandas | Exécuté |
| Gradio | Exécuté |
| SDK Groq | Exécuté |
| Véritable modèle MiniLM | Non exécuté |
| Véritables appels Groq | Non exécutés |

Deux scénarios ont été testés :

**Scénario A : sans clé API**

- 16 tests réussis.
- 0 échec.
- 2 tests non vérifiés.

**Scénario B : avec un faux LLM extractif**

- 18 tests réussis sur 18.

Le second scénario vérifie la logique du pipeline, mais ne permet pas de conclure sur la qualité des réponses produites par le véritable modèle Groq.

### Vérifications restantes

Les éléments suivants doivent être testés dans Google Colab :

- La qualité des embeddings avec le véritable modèle MiniLM.
- La pertinence des passages récupérés.
- La qualité des réponses avec Groq.
- La précision des sources affichées.
- Le comportement face aux questions sans réponse dans les documents.

Le notebook contient également une évaluation sur 7 questions, avec des résultats et des métriques calculés à partir des exécutions réelles.

## 9. Limites du projet

Le système présente certaines limites :

- Pas de reconnaissance OCR pour les PDF scannés.
- L'index FAISS est conservé uniquement en mémoire.
- L'index est partagé dans la session Colab.
- Les performances dépendent du modèle d'embedding utilisé.
- Le jeu d'évaluation contient seulement 7 questions.
- Le seuil de similarité n'est pas encore calibré.
- La vérification automatique des réponses repose principalement sur des mots-clés.

## 10. Structure du projet

```text
Intelligent-PDF-RAG/
│
├── Intelligent_PDF_QA_RAG.ipynb
│
├── demo_atlas_robotics.pdf
│
├── demo_politique_cybersecurite.pdf
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

## 11. Améliorations possibles

Plusieurs améliorations peuvent être envisagées :

- Utiliser un modèle d'embedding multilingue pour améliorer les recherches en français.
- Ajouter l'OCR pour les documents scannés.
- Comparer plusieurs tailles de chunks et valeurs d'overlap.
- Calculer Precision@k, Recall@k et MRR.
- Améliorer l'évaluation automatique des réponses.
- Ajouter une API HTTP avec FastAPI.
- Permettre la sauvegarde persistante de l'index.

## 12. Conclusion

Ce projet met en œuvre un système complet de **Retrieval-Augmented Generation (RAG)** pour la recherche d'informations dans des documents PDF.

Il combine l'extraction documentaire, le traitement du texte, les embeddings, la recherche vectorielle avec FAISS et la génération de réponses avec Groq API.

L'objectif est de fournir des réponses contextualisées, accompagnées de leurs sources, tout en limitant les hallucinations des modèles de langage.

---

**Technologies principales :** Python · Sentence Transformers · FAISS · Groq API · PyMuPDF · Gradio · Google Colab
