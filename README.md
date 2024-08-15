# RAG MBA ChatBot :robot:
**Ce repository contient le code source pour le "RAG MBA ChatBot", une application sophistiquée utilisant la technologie de recherche augmentée par génération de langage pour répondre aux questions complexes. Le chatbot est spécialement conçu pour traiter et comprendre des documents PDF, permettant des interactions dynamiques basées sur le contenu spécifique des documents chargés.**

## :sparkles: Fonctionnalités principales
- **Chargement et Traitement de PDF** : Utilisation de `PDFPlumberLoader` pour extraire le texte des documents PDF et de `SemanticChunker` pour diviser les textes en segments significatifs.
- **Embedding et Vectorisation** : Emploi des embeddings de `HuggingFace` pour transformer le texte en vecteurs, et de `FAISS` pour une indexation efficace permettant des recherches rapides.
- **Chaîne de traitement de questions-réponses** : Combinaison des techniques de récupération d'informations et de génération de réponses à l'aide de modèles préentraînés (`Ollama` basé sur le modèle `llama3`).
- **Interface utilisateur interactive** : Intégration avec `Gradio` pour une interaction simple et conviviale, permettant aux utilisateurs de charger des documents et de poser des questions directement à travers l'interface.

## :computer: Technologies utilisées
- ![LangChain](https://img.shields.io/badge/LangChain-007ACC?style=for-the-badge&logo=LangChain&logoColor=white)
- ![Hugging Face](https://img.shields.io/badge/Hugging_Face-F7931E?style=for-the-badge&logo=HuggingFace&logoColor=white)
- ![FAISS](https://img.shields.io/badge/FAISS-0052CC?style=for-the-badge&logo=FAISS&logoColor=white)
- ![Gradio](https://img.shields.io/badge/Gradio-FF4785?style=for-the-badge&logo=Gradio&logoColor=white)
- ![PDFPlumber](https://img.shields.io/badge/PDFPlumber-0769AD?style=for-the-badge&logo=PDFPlumber&logoColor=white)

## :rocket: Commencer
Pour commencer avec ce projet, suivez les instructions ci-dessous pour configurer votre environnement et lancer l'application.

### Prérequis
Assurez-vous d'avoir Python installé sur votre machine, ainsi que les packages nécessaires listés dans `requirements.txt`.

### Installation
Clonez ce dépôt en utilisant :
https://github.com/amin-chaabani/RAG-Application
