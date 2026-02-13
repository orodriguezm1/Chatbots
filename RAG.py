# ======================================================================
# RAG
# ======================================================================
# Este script implementa el flujo completo de un RAG:
#   - Lee una base Deep Lake existente.
#   - Recupera fragmentos más similares y reordena por similitud del coseno.
#   - Usa el fragmento más relevante como contexto para el LLM.
#   - Añade memoria conversacional basada en resumen (solo durante la sesión).
# ======================================================================

import os
import json
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

# Evita que Transformers use TensorFlow innecesariamente
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage


# ----------------------------------------------------------------------
# CONFIGURACIÓN Y UTILIDADES
# ----------------------------------------------------------------------
def load_config(path: str) -> Dict:
    """Carga el archivo JSON de configuración."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la similitud del coseno entre dos vectores."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_by_explicit_cosine(
    embeddings: HuggingFaceEmbeddings,
    query: str,
    documents_text: List[str],
) -> List[Tuple[int, float]]:
    """Ordena los documentos por similitud de coseno explícita."""
    q_vec = np.array(embeddings.embed_query(query), dtype=np.float32)
    d_vecs = np.array(embeddings.embed_documents(documents_text), dtype=np.float32)
    scores = [(i, cosine_similarity(q_vec, d)) for i, d in enumerate(d_vecs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ----------------------------------------------------------------------
# CARGA DE COMPONENTES
# ----------------------------------------------------------------------
def load_embeddings(cfg: Dict) -> HuggingFaceEmbeddings:
    """Carga el modelo de embeddings según la configuración."""
    emb_cfg = cfg["embedding"]
    model_name = emb_cfg["model_name"]
    device = emb_cfg.get("device", "cpu")
    normalize_embeddings = bool(emb_cfg.get("normalize_embeddings", True))
    print(f"Cargando embeddings: {model_name} (device={device})")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )


def open_deeplake(cfg: Dict, embeddings: HuggingFaceEmbeddings) -> DeepLake:
    """Abre la base Deep Lake ya existente en modo lectura."""
    dl_cfg = cfg["deeplake"]
    dataset_path = os.path.expanduser(dl_cfg["dataset_path"])
    read_only = bool(dl_cfg.get("read_only", True))

    print(f"Abrir Deep Lake en: {dataset_path} (read_only={read_only})")
    return DeepLake(dataset_path=dataset_path, embedding=embeddings, read_only=read_only)


def load_llm(cfg: Dict) -> ChatOpenAI:
    """Inicializa el modelo de lenguaje conectado a OpenRouter."""
    llm_cfg = cfg.get("llm", {})
    api_base = llm_cfg.get("api_base", "https://openrouter.ai/api/v1")
    model_name = llm_cfg.get("model_name", "mistralai/mistral-7b-instruct")
    temperature = float(llm_cfg.get("temperature", 0.7))
    max_tokens = llm_cfg.get("max_tokens", None)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en el entorno (.env)")

    print(f"Cargando LLM: {model_name} (temperature={temperature})")
    return ChatOpenAI(
        openai_api_base=api_base,
        openai_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ----------------------------------------------------------------------
# RECUPERACIÓN Y RE-RANKING
# ----------------------------------------------------------------------
def retrieve_top_by_cosine(
    db: DeepLake,
    embeddings: HuggingFaceEmbeddings,
    query: str,
    k: int,
) -> Tuple[Optional[str], Optional[Dict], List[Tuple[str, Dict, float]]]:
    """Recupera top-k fragmentos y ordena explícitamente por similitud del coseno."""
    candidates = db.similarity_search(query, k=k)
    if not candidates:
        return None, None, []

    texts = [d.page_content for d in candidates]
    metas = [d.metadata or {} for d in candidates]

    ranking = rank_by_explicit_cosine(embeddings, query, texts)
    ranked_triplets = [(texts[i], metas[i], score) for i, score in ranking]

    best_text, best_meta, _ = ranked_triplets[0]
    return best_text, best_meta, ranked_triplets


# ----------------------------------------------------------------------
# PROMPTING Y RESPUESTA
# ----------------------------------------------------------------------
def build_contextual_prompt(system_instruction: str, context_text: str, user_question: str) -> str:
    """Construye el prompt contextual para el modelo."""
    template = (
        "{system}\n\n"
        "Usa exclusivamente el siguiente CONTEXTO para responder. "
        "Si la respuesta no está en el contexto, indica que no hay suficiente información.\n\n"
        "CONTEXTO:\n"
        "-----------------------------------\n"
        "{context}\n"
        "-----------------------------------\n\n"
        "PREGUNTA:\n"
        "{question}\n\n"
        "Respuesta:"
    )
    return template.format(system=system_instruction, context=context_text, question=user_question)


# ----------------------------------------------------------------------
# MODO INTERACTIVO CON MEMORIA
# ----------------------------------------------------------------------
def interactive_rag(cfg: Dict) -> None:
    """Bucle interactivo de RAG con memoria por resumen."""
    warnings.filterwarnings("ignore")

    embeddings = load_embeddings(cfg)
    db = open_deeplake(cfg, embeddings)
    llm = load_llm(cfg)

    k = int(cfg.get("retrieval", {}).get("k", 5))
    system_instruction = cfg.get(
        "system_instruction",
        "Eres un profesor experto. Responde con precisión y claridad.",
    )

    # Configuración de memoria
    memory_cfg = cfg.get("memory", {})
    memory = None
    conversation = None

    if memory_cfg.get("enabled", False) and memory_cfg.get("type") == "summary":
        summary_model = memory_cfg.get("summary_model", llm.model_name)
        max_len = memory_cfg.get("max_summary_length", 1000)
        memory = ConversationSummaryMemory(llm=llm, max_token_limit=max_len)
        conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
        print(f"Memoria conversacional (resumen) activada con modelo: {summary_model}")
    else:
        print("Memoria desactivada (modo sin contexto).")

    print("\nRAG listo. Escribe tu pregunta. Para terminar: 'salir'.\n")

    while True:
        query = input("Pregunta: ").strip()
        if query.lower() in {"salir", "exit", "quit"}:
            print("Fin de la sesión.")
            break
        if not query:
            print("Escribe una pregunta válida.")
            continue

        try:
            best_text, best_meta, ranked = retrieve_top_by_cosine(db, embeddings, query, k=k)
            if best_text is None:
                print("No se encontraron resultados en la base.")
                continue

            print("\nTop 3 resultados por similitud del coseno:")
            for i, (t, m, s) in enumerate(ranked[:3], start=1):
                page = m.get("page", "")
                src = m.get("source", "")
                print(f"[{i}] cos={s:.4f} | página={page} | fuente={src}")
            print("-" * 80)

            # Generar la respuesta con o sin memoria
            if conversation:
                combined_input = (
                    f"{system_instruction}\n"
                    f"Contexto recuperado:\n{best_text}\n\n"
                    f"Usuario: {query}"
                )
                answer = conversation.run(combined_input)
            else:
                prompt = build_contextual_prompt(system_instruction, best_text, query)
                result = llm.invoke([HumanMessage(content=prompt)])
                answer = result.content.strip() if hasattr(result, "content") else str(result)

            print("Respuesta del asistente:")
            print(answer)
            print("-" * 80)

        except Exception as e:
            print(f"Error durante la consulta: {e}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG desde Deep Lake con memoria conversacional.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Ruta al archivo de configuración JSON.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    interactive_rag(cfg)


if __name__ == "__main__":
    main()
