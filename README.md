# Smartdoc-ai 🤖📄

Smartdoc-ai est un assistant RAG (Retrieval-Augmented Generation) multimodal capable de traiter des documents texte et audio. Il a été conçu pour indexer, transcrire, analyser et interroger des contenus complexes via une interface CLI. Ce projet s'appuie sur des technologies avancées telles que FAISS, Whisper, Langchain, Sentence-Transformers et des LLMs open source (TinyLlama, Mistral).

# Objectifs du projet

Proposer un assistant capable de traiter aussi bien des podcasts que des textes.

Mettre en œuvre un système RAG complet en local.

Tester différents LLMs : GPT-4o, TinyLlama, Mistral.

Démontrer ma capacité à faire face aux difficultés (API payante, quotas, push Git, etc.).

# Architecture globale
Smartdoc-ai/
├── data/
│   ├── audio/             # Fichiers .mp3 ou .wav à transcrire
│   ├── docs/              # Transcriptions au format .txt
│   └── index/            # Index FAISS + noms de documents
├── src/
│   ├── transcription.py   # Transcription audio avec Whisper
│   ├── embed_documents.py # Vectorisation et création de l'index FAISS
│   └── rag_hf.py         # Pipeline RAG avec LLM open-source
├── .env                   # Clés API (OpenAI, HuggingFace)
└── README.md

# Fonctionnalités principales

1. Transcription audio (Whisper)

Lecture des fichiers audio/*.mp3

Transcription avec Whisper (base) ou medium

Sauvegarde en .txt dans data/docs/

2. Indexation vectorielle (FAISS + SBERT)

Embedding des textes avec all-MiniLM-L6-v2

Stockage des vecteurs dans un index FAISS

Sauvegarde des noms pour retrouver les contextes

3. RAG avec LLM local (TinyLlama / Mistral)

Interrogation par l'utilisateur via la console

Requête vectorielle sur les documents

Construction d'un prompt contextuel

Génération de réponses pertinentes

# LLMs testés:
               
Modèle  	     Langue	Poids  	Temps de réponse	  Support du français 	     Remarques
GPT-4o (API)  	🇫🇷/🇺🇸    	         cloud 	   Rapide    	     Excellent 	Très pertinent mais dépend d'un quota payant
TinyLlama-1.1B	🇺🇸 uniquement  	 ~1.1B  	   Moyen      	 Limité      	Léger, rapide, mais mauvaise compréhension du français
Mistral-7B     	       🇫🇷/🇺🇸      	~7B    	  Lent (CPU)      	  Bon support   	    Recommandé si GPU ou version quantisée disponible


# Problèmes rencontrés et solutions apportées: 

Problème       	                                 Solution apportée
Quota API GPT-4o dépassé     	                 Migration vers LLM open-source (TinyLlama puis Mistral)
Problèmes de push (error 408, broken pipe)   	 Nettoyage Git, suppression fichiers lourds, push via SSH
Prompt trop long pour les LLMs       	         Limitation à 2500 caractères et top 3 documents (FAISS)
Modèle ne supporte pas le français    	         Test de plusieurs LLMs jusqu'à Mistral pour support natif

# Exécution du projet (en local) :

1-Cloner le repo : git clone git@github.com:Lbenzid/Smartdoc-ai.git
cd Smartdoc-ai


2-Créer un environnement Python:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


3-Configurer les clés API dans un .env:
OPENAI_API_KEY=sk-...
HUGGINGFACEHUB_API_TOKEN=hf_...

4-Transcrire les audios: python src/transcription.py

5-Indexer les documents: python src/embed_documents.py

6-Lancer le RAG avec LLM local: python src/rag_hf.py

# Exemples de questions posées : 
Quel est le thème principal du document ?

Quels sont les risques évoqués liés à l’IA dans l’art ?

Peux-tu résumer ce podcast en 3 phrases ?

Le document fait-il référence à des lois ou entreprises ?

# Perspectives d'évolution

Interface web avec Gradio ou Streamlit

Support multi-documents et multi-langues

Amélioration du prompt engineering

Ajout de la recherche sémantique hybride (BM25 + embeddings)


Projet open source réalisé par @Lbenzid ✨

