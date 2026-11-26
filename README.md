#  Intelligent PDF Question Answering System Using RAG

Application de questions-réponses sur des documents PDF, fondée sur la **Retrieval-Augmented Generation (RAG)** : l'utilisateur importe un ou plusieurs PDF, pose une question, et reçoit une réponse **basée uniquement sur ces documents**, avec le nom du PDF et le numéro de page des sources.

## 1. Problème traité

Un LLM seul ne connaît pas vos documents et peut inventer des réponses (« hallucinations »). Ici, le LLM ne reçoit que les 3 à 5 passages les plus pertinents extraits de vos PDF, avec l'ordre de ne répondre qu'à partir d'eux. Si l'information n'existe pas, le système répond :

> « Je ne trouve pas cette information dans les documents fournis. »

## 2. Architecture RAG

```
 PDF ──► PyMuPDF ──► nettoyage ──► chunks (600 car., chevauchement 100)
                                        │
                      Sentence Transformers (all-MiniLM-L6-v2, 384 dim.)
                                        ▼
                         index FAISS (IndexFlatIP, vecteurs normalisés)
                                        ▲
 Question ──► embedding ──► top 3-5 passages ──► Groq API (LLM) ──► réponse + sources
```

## 3. Technologies

| Rôle | Technologie |
|---|---|
| LLM | Groq API (`openai/gpt-oss-120b`, repli sur `openai/gpt-oss-20b`, puis `qwen/qwen3.6-27b`) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Recherche vectorielle | FAISS (`faiss-cpu`) |
| Extraction PDF | PyMuPDF |
| Interface | Gradio |
| Environnement | Google Colab |

**À propos du modèle Groq** : d'après la documentation de Groq, `llama-3.3-70b-versatile` et `llama-3.1-8b-instant` ont été arrêtés le 16/08/2026, avec `openai/gpt-oss-120b` et `qwen/qwen3.6-27b` comme remplaçants recommandés. Groq retirant régulièrement des modèles, le notebook **liste les modèles disponibles sur votre compte au lancement**, choisit le premier de la liste `GROQ_MODEL_CANDIDATES`, et bascule automatiquement sur le suivant si un modèle est refusé. Pour changer de modèle, modifiez cette liste dans la cellule 3.

## 4. Fonctionnement

1. **Extraction** : PyMuPDF lit chaque page ; le nom du fichier et le numéro de page sont conservés. Les fichiers invalides, corrompus, protégés par mot de passe, vides ou sans texte (PDF scannés, sans OCR) sont signalés sans faire planter l'application.
2. **Prétraitement** : nettoyage du texte (ligatures, mots coupés, espaces), puis découpage **page par page** en chunks de **600 caractères maximum** avec **100 caractères de chevauchement**, coupés de préférence sur un espace.
3. **Indexation** : embeddings normalisés (norme = 1) → index FAISS. Le produit scalaire équivaut alors au cosinus. Un document est identifié par le **SHA-256 de son contenu** : le même PDF n'est jamais indexé deux fois, même renommé.
4. **Recherche** : la question est transformée en embedding ; les `TOP_K = 4` passages les plus proches sont récupérés (valeur modifiable entre 3 et 5).
5. **Génération** : seuls ces passages sont envoyés à Groq, avec des consignes strictes (réponse uniquement depuis le contexte, phrase d'absence exacte sinon, résistance aux instructions cachées dans les PDF). Le LLM indique les extraits réellement utilisés ; **seuls ceux-là sont affichés comme sources** (fichier + page).
6. **Garde-fou** : si le meilleur passage a une similarité < `MIN_SCORE` (0,15), la réponse d'absence est renvoyée sans appeler le LLM. Ce seuil est volontairement bas et **n'a pas été calibré** sur vos documents.

Les erreurs d'API (clé invalide 401, limite de requêtes 429, délai dépassé, connexion, modèle retiré, erreur serveur) sont converties en messages clairs ; aucune exception n'est affichée à l'utilisateur.

## 5. Installation et configuration

### Dans Google Colab (recommandé)
1. Ouvrez `Intelligent_PDF_QA_RAG.ipynb` dans Colab (*Fichier → Importer*).
2. Créez une clé gratuite sur <https://console.groq.com/keys>.
3. Dans Colab : icône 🔑 **Secrets** → *Ajouter un secret* → nom **`GROQ_API_KEY`**, valeur = votre clé → activez **l'accès au notebook**.
4. *Exécution → Tout exécuter*.

La clé n'est jamais écrite dans le notebook ni affichée : elle est lue par `userdata.get("GROQ_API_KEY")`.

### En local (facultatif)
```bash
pip install -r requirements.txt
export GROQ_API_KEY="votre_clé"      # hors Colab, la clé est lue dans l'environnement
```
Puis ouvrez le notebook avec Jupyter (la 1re cellule `!pip install` peut être ignorée).

## 6. Étapes d'exécution

Le notebook suit cet ordre : présentation → installation → imports et paramètres → configuration Groq → modèle d'embedding → extraction PDF → chunking → embeddings → index FAISS → recherche → génération → pipeline complet → tests unitaires → **tests avec de vrais PDF** → évaluation → interface Gradio → **lancement (dernière cellule)**.

Dans l'interface :
1. Importez un ou plusieurs PDF → **Indexer les documents** (barre de progression + statut par fichier).
2. Saisissez une question → **Poser la question** → réponse + section **Sources utilisées**.
3. **Réinitialiser les documents indexés** vide l'index.

Les PDF de démonstration (`demo_pdfs/`) sont créés par le notebook lui-même (section 14) ; ils sont aussi fournis ici. Ils décrivent des entreprises **fictives** : `demo_atlas_robotics.pdf` (robotique agricole) et `demo_politique_cybersecurite.pdf` (politique de sécurité), 3 pages chacun, avec des chiffres inventés et vérifiables.

## 7. Tests et résultats

Le notebook contient 7 tests unitaires et les tests 1 à 8 demandés (extraction, chunking, embeddings, FAISS, question avec réponse, question absente, plusieurs PDF, erreurs). Un test n'affiche ✅ que si toutes ses vérifications passent ; s'il ne peut pas s'exécuter (clé absente, API injoignable), il affiche ⏭️ **« non vérifié »**, jamais ✅.

### Ce qui a été réellement exécuté lors de la livraison

Le notebook a été exécuté cellule par cellule dans un environnement de développement **sans accès à Hugging Face ni à l'API Groq**. Deux scénarios ont donc été joués, avec des **remplaçants explicitement identifiés** :

| Élément | Réel | Remplacé par |
|---|---|---|
| PyMuPDF, FAISS, pandas, Gradio 6.32 | ✅ réels | — |
| SDK `groq` (gestion des exceptions) | ✅ réel | — |
| Modèle `all-MiniLM-L6-v2` | ❌ **non exécuté** | faux embedder (hachage de mots/trigrammes, 384 dim.) |
| Appels à l'API Groq | ❌ **non exécutés** | scénario A : aucun appel ; scénario B : faux LLM « extractif » |

**Scénario A — sans clé API** (16 ✅, 0 ❌, 2 ⏭️ sur 18) : les 7 tests unitaires, les tests 1, 2, 3, 4, 7 et 8 (a, b, c, d) passent. Les tests 5 et 6, qui exigent un vrai LLM, sont correctement marqués ⏭️ « non vérifié ».
**Scénario B — faux LLM** (18 ✅ sur 18) : valide la **logique** des tests 5 et 6, de l'analyse des sources et de l'évaluation, mais **ne prouve rien sur la qualité de Groq**.

Constatations concrètes (réelles, avec le faux embedder) :
- Extraction : 3 pages par PDF, accents correctement extraits. Chunking : 6 chunks par PDF, taille de 162 à 599 caractères (jamais > 600).
- Index : 12 vecteurs pour les 2 PDF ; le ré-import d'un même PDF est refusé (« doublon »).
- Erreurs : fichier texte déguisé en PDF, fichier vide, PDF sans page/corrompu, PDF sans texte, PDF protégé par mot de passe et fichier inexistant sont tous rejetés avec un message précis. Les erreurs 429, 401, 500, délai dépassé et serveur injoignable renvoient un message en français (ces exceptions sont simulées avec de vrais objets d'erreur du SDK ; l'une est une vraie panne de connexion).
- Gradio : l'application a été lancée et ses boutons exercés via le client Gradio (indexation, doublon, fichier invalide, question, question absente, question vide, réinitialisation).

### Ce qui reste à vérifier dans Colab (non exécuté par l'auteur)
- Les résultats avec le **vrai modèle MiniLM** : notamment le rang de la bonne source (test 7 et évaluation). `all-MiniLM-L6-v2` est surtout entraîné sur l'anglais ; les PDF de démo sont en français, donc la récupération peut être moins nette qu'avec le faux embedder. Si c'est le cas, remplacez `EMBEDDING_MODEL_NAME` par `paraphrase-multilingual-MiniLM-L12-v2` (la dimension est lue automatiquement).
- Les tests **5 et 6** et la **qualité des réponses** avec le vrai LLM Groq.

Après exécution, la cellule « Bilan des tests » et la section 15 (évaluation) affichent vos **résultats réels** ; le tableau d'évaluation (7 questions, réponses attendues/obtenues, sources récupérées et rang) et les métriques (récupération, MRR, réponses correctes) sont calculés uniquement à partir de ces données. Ils ne sont pas reproduits ici pour ne pas présenter comme réels des chiffres que je n'ai pas obtenus.

## 8. Limites connues
- Pas d'OCR : un PDF scanné (images) est signalé « sans texte extractible ».
- L'index est en mémoire et partagé par tous les utilisateurs de la session (adapté à Colab, pas à un déploiement multi-utilisateurs). Il est perdu à la fermeture du notebook.
- Le jeu d'évaluation est petit (7 questions, documents courts) : il donne une indication, pas une mesure statistique.
- La vérification automatique des réponses est basique (mots-clés) ; relisez les réponses affichées.
- Dans Colab, Gradio crée un lien public temporaire : ne le partagez pas, il utilise votre clé Groq.

## 9. Dépannage
- **« Clé GROQ_API_KEY introuvable »** : vérifiez le nom exact du secret et que l'accès au notebook est activé.
- **Erreur 429** : limite de requêtes Groq atteinte ; patientez quelques secondes.
- **Erreur d'import `numpy`/`faiss` après l'installation** : *Exécution → Redémarrer la session*, puis relancez.

## 10. Fichiers
- `Intelligent_PDF_QA_RAG.ipynb` : notebook complet
- `requirements.txt` : dépendances
- `demo_pdfs/` : deux PDF de démonstration fictifs
