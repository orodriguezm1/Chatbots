import os
import json
import re
import warnings
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory

warnings.filterwarnings("ignore")

# 🔐 Cargar variables desde el archivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ No se encontró OPENAI_API_KEY en el archivo .env")

# 🤖 Configuración del modelo vía OpenRouter
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name="mistralai/mistral-7b-instruct",
    temperature=0.2,
)

RESULT_FILE = "resultado.json"

# 🧠 Memoria (LangChain): guarda las últimas K interacciones (usuario/bot)
memory = ConversationBufferWindowMemory(k=10, return_messages=True)


# ----------------------------
# Utilidades
# ----------------------------
def safe_json_extract(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def ask_menu(
    question: str,
    options: list[str],
    recommended: str | None = None,
    recommendation_reason: str | None = None,
    allow_recommend_phrases: bool = True,
) -> str:
    """
    Menú numerado con opción de recomendación.
    - Si el usuario escribe "no lo sé", "recomienda", "cualquiera", etc.,
      devuelve la opción recomendada.
    - Acepta también escribir el nombre exacto de la opción.
    - Si hay recommended + recommendation_reason, lo explica en lenguaje sencillo.
    """
    recommend_phrases = {
        "no lo se", "no lo sé", "nose", "no sé", "ni idea",
        "cualquiera", "recomendado", "recomienda", "recomendacion", "recomendación",
        "elige tu", "elige tú", "tu decides", "tú decides", "lo que sea"
    }

    # fallback si no pasan recommended o no es válido
    if recommended not in options:
        recommended = options[0]
        recommendation_reason = None  # si el recomendado cambió, mejor no inventar razón

    # Explicación corta antes del menú (una sola vez por pregunta)
    if recommended and recommendation_reason:
        print(f"\n🤖 Recomendación: {recommended}")
        print(f"🤖 ¿Por qué? {recommendation_reason}")

    while True:
        print("\n" + question)
        for i, opt in enumerate(options, start=1):
            tag = " ⭐ recomendado" if opt == recommended else ""
            print(f"  {i}) {opt}{tag}")

        user = input("👉 Elige un número (o escribe 'recomendado'): ").strip().lower()

        if user in ["salir", "exit", "quit"]:
            raise SystemExit

        # "no lo sé" -> recomendado
        if allow_recommend_phrases and user in recommend_phrases:
            print(f"🤖 Ok, uso el recomendado: {recommended}")
            return recommended

        # número
        if user.isdigit():
            idx = int(user)
            if 1 <= idx <= len(options):
                return options[idx - 1]

        # permitir escribir la opción
        for opt in options:
            if user == opt.lower():
                return opt

        print("❌ Opción inválida. Escribe un número del menú o 'recomendado'.")


def ask_yes_no(question: str) -> bool:
    while True:
        ans = input(question + " (sí/no): ").strip().lower()
        if ans in ["sí", "si", "s"]:
            return True
        if ans in ["no", "n"]:
            return False
        print("❌ Responde 'sí' o 'no'.")


def ask_int_in_range(question: str, min_v: int, max_v: int) -> int:
    while True:
        user = input(f"{question} ({min_v}-{max_v}): ").strip()
        if user.isdigit():
            v = int(user)
            if min_v <= v <= max_v:
                return v
        print("❌ Valor inválido. Intenta de nuevo.")


def get_recent_memory_messages():
    """Devuelve lista de mensajes (HumanMessage/AIMessage) desde la memoria."""
    vars_ = memory.load_memory_variables({})
    return vars_.get("history", [])


# ----------------------------
# Router con memoria (propuesta + explicación + pregunta)
# ----------------------------
def propose_type_with_explanation() -> dict:
    """
    Usa el historial (memoria LangChain) para proponer:
      - tipo: cluster|series_tiempo|preguntar
      - explicacion: resumen sencillo de lo entendido
      - pregunta: si falta info, una sola pregunta
      - confianza: 0..1
    """
    system = SystemMessage(
        content=(
            "Eres un asistente que decide qué configuración JSON de ML construir.\n"
            "Responde SOLO con un JSON válido (sin texto adicional).\n\n"
            "Clases:\n"
            '- "cluster": si el usuario quiere AGRUPAR cosas similares (segmentación, clustering).\n'
            '- "series_tiempo": si el usuario quiere PREDECIR valores futuros en el tiempo (forecast).\n'
            '- "preguntar": si no está claro; devuelve UNA pregunta corta.\n\n'
            "Además, incluye una explicación MUY sencilla para que el usuario confirme."
        )
    )

    prompt = HumanMessage(
        content=(
            "Con base en la conversación, decide el tipo.\n"
            "Formato obligatorio:\n"
            '{\n'
            '  "tipo": "cluster|series_tiempo|preguntar",\n'
            '  "explicacion": "En lenguaje sencillo, qué entendiste que quiere el usuario",\n'
            '  "pregunta": "Solo si tipo=preguntar, si no vacio",\n'
            '  "confianza": 0.0\n'
            "}\n"
        )
    )

    history = get_recent_memory_messages()
    resp = llm.invoke([system] + history + [prompt])
    content = getattr(resp, "content", str(resp))
    data = safe_json_extract(content)

    if not isinstance(data, dict):
        return {
            "tipo": "preguntar",
            "explicacion": "No estoy seguro de si quieres agrupar cosas similares o predecir valores en el tiempo.",
            "pregunta": "¿Tu objetivo es agrupar elementos parecidos (clustering) o predecir valores futuros (series de tiempo)?",
            "confianza": 0.0,
        }

    tipo = data.get("tipo", "preguntar")
    if tipo not in ["cluster", "series_tiempo", "preguntar"]:
        tipo = "preguntar"

    explicacion = str(data.get("explicacion", "")).strip()
    pregunta = str(data.get("pregunta", "")).strip() if tipo == "preguntar" else ""
    try:
        confianza = float(data.get("confianza", 0.0))
    except Exception:
        confianza = 0.0

    return {
        "tipo": tipo,
        "explicacion": explicacion,
        "pregunta": pregunta,
        "confianza": max(0.0, min(1.0, confianza)),
    }


# ----------------------------
# Plantillas JSON y opciones (valores limitados)
# ----------------------------
CLUSTER_SCHEMA = {
    "tipo": "cluster",
    "algoritmo": None,         # kmeans | dbscan | aglomerativo
    "num_clusters": None,      # 2-10 o "auto"
    "escalado": None,          # zscore | minmax | ninguno
    "metrica": None,           # euclidean | manhattan | cosine
    "objetivo": None,          # segmentacion_clientes | agrupacion_documentos | deteccion_anomalias
}

SERIES_SCHEMA = {
    "tipo": "series_tiempo",
    "modelo": None,            # arima | prophet | lstm
    "horizonte": None,         # 1-30
    "frecuencia": None,        # diaria | semanal | mensual
    "metrica_error": None,     # mae | rmse | mape
}

CLUSTER_OPTIONS = {
    "algoritmo": ["kmeans", "dbscan", "aglomerativo"],
    "num_clusters": ["auto"] + [str(i) for i in range(2, 11)],
    "escalado": ["zscore", "minmax", "ninguno"],
    "metrica": ["euclidean", "manhattan", "cosine"],
    "objetivo": ["segmentacion_clientes", "agrupacion_documentos", "deteccion_anomalias"],
}

SERIES_OPTIONS = {
    "modelo": ["arima", "prophet", "lstm"],
    "frecuencia": ["diaria", "semanal", "mensual"],
    "metrica_error": ["mae", "rmse", "mape"],
}


# ----------------------------
# Flujos guiados para llenar JSON
# ----------------------------
def fill_cluster():
    cfg = dict(CLUSTER_SCHEMA)

    print("\n🧩 Configuración de CLUSTERING (agrupar elementos similares).")

    cfg["objetivo"] = ask_menu(
        "¿Qué quieres lograr? (si no sabes, elige recomendado)",
        CLUSTER_OPTIONS["objetivo"],
        recommended="segmentacion_clientes",
        recommendation_reason="Es el caso más común: separar usuarios/elementos en grupos parecidos para entender perfiles."
    )

    # Recomendación depende del objetivo (regla simple)
    if cfg["objetivo"] == "deteccion_anomalias":
        rec_alg = "dbscan"
        rec_k = "auto"
        alg_reason = "DBSCAN suele ayudar a separar puntos raros (anómalos) sin exigir un número fijo de grupos."
        k_reason = "En anomalias es mejor no forzar un K fijo."
    else:
        rec_alg = "kmeans"
        rec_k = "3"
        alg_reason = "KMeans es una opción simple y estable para empezar cuando quieres grupos claros."
        k_reason = "3 es un punto de partida típico; luego se ajusta con resultados."

    cfg["algoritmo"] = ask_menu(
        "¿Qué algoritmo prefieres? (si no sabes, elige recomendado)",
        CLUSTER_OPTIONS["algoritmo"],
        recommended=rec_alg,
        recommendation_reason=alg_reason
    )

    cfg["num_clusters"] = ask_menu(
        "¿Cuántos grupos quieres? (auto intenta estimarlo)",
        CLUSTER_OPTIONS["num_clusters"],
        recommended=rec_k,
        recommendation_reason=k_reason
    )
    if cfg["num_clusters"] != "auto":
        cfg["num_clusters"] = int(cfg["num_clusters"])

    cfg["escalado"] = ask_menu(
        "¿Cómo quieres normalizar/escalar tus datos?",
        CLUSTER_OPTIONS["escalado"],
        recommended="zscore",
        recommendation_reason="Es una normalización estándar que ayuda cuando las variables están en escalas diferentes."
    )

    cfg["metrica"] = ask_menu(
        "¿Qué métrica de distancia usar?",
        CLUSTER_OPTIONS["metrica"],
        recommended="euclidean",
        recommendation_reason="Es la distancia más común para datos numéricos y funciona bien como primera opción."
    )

    return cfg




def fill_series():
    cfg = dict(SERIES_SCHEMA)

    print("\n📈 Configuración de SERIES DE TIEMPO (predecir futuro con historial).")

    cfg["frecuencia"] = ask_menu(
        "¿Con qué frecuencia están tus datos?",
        SERIES_OPTIONS["frecuencia"],
        recommended="diaria",
        recommendation_reason="La mayoría de datos operativos vienen a diario; si no estás seguro, esta suele ser la más común."
    )

    cfg["horizonte"] = ask_int_in_range(
        "¿Cuántos pasos hacia el futuro quieres predecir?",
        1,
        30
    )

    if cfg["frecuencia"] in ["diaria", "semanal"]:
        rec_modelo = "prophet"
        model_reason = "Prophet suele funcionar bien como primera opción y maneja estacionalidades de forma sencilla."
    else:
        rec_modelo = "arima"
        model_reason = "ARIMA es un buen modelo clásico para empezar cuando los datos son más agregados."

    cfg["modelo"] = ask_menu(
        "¿Qué modelo quieres usar? (si no sabes, elige recomendado)",
        SERIES_OPTIONS["modelo"],
        recommended=rec_modelo,
        recommendation_reason=model_reason
    )

    cfg["metrica_error"] = ask_menu(
        "¿Cómo medimos qué tan bueno es el modelo?",
        SERIES_OPTIONS["metrica_error"],
        recommended="mae",
        recommendation_reason="MAE es fácil de interpretar: en promedio, cuánto te equivocas en unidades reales."
    )

    return cfg




def explain_solution(cfg: dict) -> str:
    if cfg["tipo"] == "cluster":

        if cfg["num_clusters"] == "auto":
            grupos_txt = "un número automático de grupos"
        else:
            grupos_txt = f"{cfg['num_clusters']} grupos"

        return (
            "Voy a agrupar tus datos en grupos de elementos parecidos. "
            f"Usaré el algoritmo '{cfg['algoritmo']}' y {grupos_txt}. "
            f"Primero escalaré los datos con '{cfg['escalado']}' "
            f"y mediré similitud con distancia '{cfg['metrica']}'. "
            f"El objetivo es '{cfg['objetivo']}'."
        )

    return (
        "Voy a entrenar un modelo para predecir valores futuros usando tu historial. "
        f"Usaré '{cfg['modelo']}', con datos de frecuencia '{cfg['frecuencia']}', "
        f"y predeciré {cfg['horizonte']} paso(s) hacia adelante. "
        f"Evaluaré el desempeño con '{cfg['metrica_error']}'."
    )


# ----------------------------
# Chat principal (entender -> confirmar -> llenar -> guardar)
# ----------------------------
print("💬 Asistente para crear configuración JSON de ML (escribe 'salir' para terminar)\n")

state = "entender"
chosen_type = None

while True:
    if state == "entender":
        user_input = input("👤 Cuéntame tu problema (ej: 'segmentar clientes' o 'predecir ventas'): ").strip()
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 Hasta luego.")
            break

        # Guardar en memoria (LangChain)
        memory.chat_memory.add_user_message(user_input)

        # Intentos de aclaración/confirmación usando memoria
        for _ in range(4):
            proposal = propose_type_with_explanation()

            if proposal["tipo"] == "preguntar":
                q = proposal["pregunta"] or "¿Quieres agrupar elementos similares o predecir valores futuros en el tiempo?"
                print("\n🤖 Necesito una aclaración corta:")
                print("🤖 " + q)
                ans = input("👤 Respuesta: ").strip()
                if ans.lower() in ["salir", "exit", "quit"]:
                    print("👋 Hasta luego.")
                    raise SystemExit

                memory.chat_memory.add_user_message(ans)
                continue

            # Ya propone tipo: explicar + pedir confirmación
            explanation = proposal["explicacion"] or (
                "Entendí que quieres " +
                ("agrupar cosas similares (clustering)." if proposal["tipo"] == "cluster" else "predecir valores futuros (series de tiempo).")
            )

            print("\n🤖 Entendí esto (dime si es correcto):")
            print("🤖 " + explanation)

            ok = ask_yes_no("👉 ¿Es correcto?")
            memory.chat_memory.add_ai_message("Entendí: " + explanation)

            if ok:
                chosen_type = proposal["tipo"]
                state = "llenar"
                break

            print("\n🤖 Perfecto. Dímelo en una frase:")
            correction = input("👤 ¿Qué quieres lograr exactamente?: ").strip()
            if correction.lower() in ["salir", "exit", "quit"]:
                print("👋 Hasta luego.")
                raise SystemExit
            memory.chat_memory.add_user_message(correction)

        if state != "llenar":
            # Fallback si aún hay ambigüedad
            chosen_type = ask_menu(
                "Para no bloquearte, elige lo que más se parezca a tu objetivo:",
                ["cluster", "series_tiempo"]
            )
            state = "llenar"

    elif state == "llenar":
        if chosen_type == "cluster":
            cfg = fill_cluster()
        else:
            cfg = fill_series()

        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Listo. Generé el archivo interno: {RESULT_FILE}")
        print("\n🧠 Así se resolverá tu problema (en sencillo):")
        print("🤖 " + explain_solution(cfg) + "\n")

        break
