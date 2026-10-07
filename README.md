# RAG base - Assistente HR

Cerca tra i CV nella cartella `resumes/` il profilo più adatto a una richiesta.

## Installazione

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Configurazione

Copia `.env.example` in `.env` e inserisci i tuoi valori.
Se `EMBEDDING_API_KEY` è vuota, gli embeddings vengono calcolati in locale da ChromaDB
(la prima volta scarica un piccolo modello).

## Avvio

python rag_base.py

## Formato dei CV

File `.txt` in `resumes/`, con sezioni separate da `### `.
La prima sezione deve contenere i dati personali (nome, email, telefono).
