"""
RAG base: assistente HR che trova il CV più adatto a una richiesta.

Flusso (come nello schema del corso):
documenti -> chunk -> embeddings -> vector store (ChromaDB)
domanda -> embedding -> ricerca nel vector store -> contesto
contesto + domanda -> prompt template -> LLM -> risposta
"""
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOCUMENTS_DIR = "resumes"
COLLECTION_NAME = "CVs"
N_RESULTS = 1

SYSTEM_PROMPT = "Sei un assistente HR, specializzato nella ricerca di profili professionali"


# 1. LETTURA DEI DOCUMENTI E CHUNKING
def leggi_documenti(cartella):
    """Legge i .txt e li divide in chunk usando '### ' come separatore.

    Il primo chunk dopo '### ' (chunks[1]) contiene i dati personali:
    lo salviamo nei metadati di ogni chunk per poterlo recuperare dopo.
    """
    documents, metadatas, ids = [], [], []
    id_counter = 0

    for filename in sorted(os.listdir(cartella)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(cartella, filename), "r", encoding="utf-8") as file:
            chunks = file.read().replace("\n", ".").split("### ")

        if len(chunks) < 2:
            continue

        for chunk in chunks:
            if not chunk.isspace() and chunk != "":
                documents.append(chunk)
                metadatas.append({"source": filename, "info": chunks[1]})
                ids.append(str(id_counter))
                id_counter += 1

    return documents, metadatas, ids


# 2. EMBEDDINGS
def crea_embedding_function():
    """Embeddings OpenAI se c'è la chiave, altrimenti modello locale di ChromaDB."""
    key = os.getenv("EMBEDDING_API_KEY")
    if key:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=key,
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    print("EMBEDDING_API_KEY non impostata: uso gli embeddings locali di ChromaDB.")
    return embedding_functions.DefaultEmbeddingFunction()


# 3. VECTOR STORE
def crea_collezione(embedding_function):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    documents, metadatas, ids = leggi_documenti(DOCUMENTS_DIR)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Indicizzati {collection.count()} chunk da '{DOCUMENTS_DIR}'.")
    return collection


# 4. RETRIEVAL (vector similarity)
def cerca(collection, user_question):
    return collection.query(query_texts=[user_question], n_results=N_RESULTS)


# 5. CONTESTO E PROMPT TEMPLATE
def costruisci_prompt(results, user_question):
    source = results["metadatas"][0][0]["source"]
    info = results["metadatas"][0][0]["info"]
    paragrafo = results["documents"][0][0]

    context = (
        f"CONTESTO: nome file {source} "
        f"ecco il paragrafo piu' significativo: {paragrafo} "
        f"ricorda sempre di menzionare il nome del candidato all'inizio "
        f"e i dati personali alla fine per il contatto, ti lascio tutto qui: {info}"
    )

    return f"""Dato il seguente contesto {context} rispondi alla domanda dell'utente {user_question}
spiegando che nel file individuato c'e' il profilo piu' adatto.
Argomenta la scelta utilizzando il contenuto del testo individuato nel contesto
"""


# 6. LLM
def genera_risposta(client, prompt):
    completion = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def main():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    collection = crea_collezione(crea_embedding_function())

    print("\nEsempi: 'mi serve qualcuno per promuovere il mio prodotto'")
    print("        'mi serve qualcuno per mettere in sicurezza il mio sito web'")
    print("        'mi serve qualcuno certificato CEH'")

    while True:
        user_question = input("\nDomanda (quit per uscire): ").strip()
        if user_question.lower() in ["quit", "exit", "q"]:
            print("Ciao!")
            break
        if not user_question:
            continue

        results = cerca(collection, user_question)
        print(f"\n[Chunk trovato in: {results['metadatas'][0][0]['source']}]")

        prompt = costruisci_prompt(results, user_question)
        print("\n" + genera_risposta(client, prompt))


if __name__ == "__main__":
    main()
