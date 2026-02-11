import os
import json
import re
import warnings
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
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


def is_out_of_scope(text: str) -> bool:
    """
    Bloqueo simple: si el usuario se sale del contexto durante menús/preguntas,
    no respondemos a eso, repetimos la pregunta.
    """
    t = text.strip().lower()
    off = [
        "chiste", "jaja", "hola", "quien eres", "quién eres",
        "cuanto es", "cuánto es", "cuentame", "cuéntame",
        "dime algo", "gracias", "adios", "adiós"
    ]
    return any(x in t for x in off)


def explain_options(user_text: str, options: list[str]) -> str | None:
    """
    Si el usuario pide ayuda/explicación (ej. "explicame rmse"), devuelve texto explicativo.
    """
    u = user_text.strip().lower()
    help_triggers = [
        "explica", "explícame", "explicame", "que es", "qué es", "ayuda",
        "diferencia", "diferencias", "para que sirve", "para qué sirve",
        "cual es mejor", "cuál es mejor", "como elijo", "cómo elijo"
    ]
    if not (any(t in u for t in help_triggers) or any(opt.lower() in u for opt in options)):
        return None

    metric_expl = {
        "mae": "MAE: error promedio en unidades reales. Fácil de entender. Menor = mejor.",
        "rmse": "RMSE: como MAE pero castiga más los errores grandes. Útil si te importan mucho los picos. Menor = mejor.",
        "mape": "MAPE: error en porcentaje. Fácil de comunicar, pero falla si hay valores cercanos a 0. Menor = mejor.",
    }
    model_expl = {
        "arima": "ARIMA: modelo clásico para series; suele ir bien con series relativamente estables.",
        "prophet": "Prophet: práctico para empezar; maneja estacionalidad y festivos de forma sencilla.",
        "lstm": "LSTM: red neuronal; puede servir con muchos datos, pero es más compleja de entrenar y ajustar.",
    }
    cluster_expl = {
        "kmeans": "KMeans: agrupa creando centros; simple y buen punto de partida.",
        "dbscan": "DBSCAN: detecta grupos por densidad; útil para encontrar anomalías y no requiere K fijo.",
        "aglomerativo": "Aglomerativo: clustering jerárquico; útil para explorar estructuras de grupos.",
    }
    scale_expl = {
        "zscore": "Z-Score: centra y escala por desviación estándar; muy usado por defecto.",
        "minmax": "MinMax: escala a un rango (0-1); útil si quieres limitar rangos.",
        "ninguno": "Ninguno: no escala; solo si tus variables ya están comparables.",
    }
    metric_dist_expl = {
        "euclidean": "Euclidiana: distancia típica para datos numéricos.",
        "manhattan": "Manhattan: suma diferencias absolutas; más robusta ante ciertos outliers.",
        "cosine": "Coseno: mide ángulo/similitud; común en texto/vectores.",
    }

    expl_maps = [metric_expl, model_expl, cluster_expl, scale_expl, metric_dist_expl]

    lines = []
    for opt in options:
        key = opt.lower()
        for m in expl_maps:
            if key in m:
                lines.append(f"- {opt}: {m[key]}")
                break

    if not lines:
        return None

    return "📝 Te explico las opciones:\n" + "\n".join(lines)


def ask_menu(
    question: str,
    options: list[str],
    recommended: str | None = None,
    recommendation_reason: str | None = None,
    allow_recommend_phrases: bool = True,
) -> str:
    """
    Menú numerado con:
    - recomendado (⭐) + explicación
    - soporte a: "no lo sé", "recomendado"
    - soporte a: pedir "ayuda/explicación" sin romper el flujo
    - bloqueo a salirse de contexto: repite la pregunta
    """
    recommend_phrases = {
        "no lo se", "no lo sé", "nose", "no sé", "ni idea",
        "cualquiera", "recomendado", "recomienda", "recomendacion", "recomendación",
        "elige tu", "elige tú", "tu decides", "tú decides", "lo que sea",
        "cual recomiendas", "cuál recomiendas"
    }

    if recommended not in options:
        recommended = options[0]
        recommendation_reason = None

    if recommended and recommendation_reason:
        print(f"\n🤖 Recomendación: {recommended}")
        print(f"🤖 ¿Por qué? {recommendation_reason}")

    while True:
        print("\n" + question)
        for i, opt in enumerate(options, start=1):
            tag = " ⭐ recomendado" if opt == recommended else ""
            print(f"  {i}) {opt}{tag}")

        user = input("👉 Elige un número (o escribe 'recomendado' / 'ayuda'): ").strip().lower()

        if user in ["salir", "exit", "quit"]:
            raise SystemExit

        # Bloqueo fuera de contexto
        if is_out_of_scope(user):
            print("🤖 Solo puedo ayudarte a configurar el modelo/JSON. Para avanzar necesito una opción del menú.")
            continue

        # Ayuda/explicación
        help_text = explain_options(user, options)
        if help_text:
            print("\n" + help_text)
            print("\n👉 Ahora elige una opción del menú (o escribe 'recomendado').")
            continue

        # "no lo sé" -> recomendado
        if allow_recommend_phrases and user in recommend_phrases:
            print(f"🤖 Ok, uso el recomendado: {recommended}")
            return recommended

        # número
        if user.isdigit():
            idx = int(user)
            if 1 <= idx <= len(options):
                return options[idx - 1]

        # permitir escribir la opción exacta
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
        if is_out_of_scope(ans):
            print("🤖 Para avanzar necesito 'sí' o 'no'.")
            continue
        print("❌ Responde 'sí' o 'no'.")


def ask_int_in_range(question: str, min_v: int, max_v: int) -> int:
    while True:
        user = input(f"{question} ({min_v}-{max_v}): ").strip().lower()
        if user in ["salir", "exit", "quit"]:
            raise SystemExit
        if is_out_of_scope(user):
            print("🤖 Solo puedo ayudarte a configurar el modelo/JSON. Para avanzar necesito un número.")
            continue
        if user.isdigit():
            v = int(user)
            if min_v <= v <= max_v:
                return v
        print("❌ Valor inválido. Intenta de nuevo.")


def get_recent_memory_messages():
    vars_ = memory.load_memory_variables({})
    return vars_.get("history", [])


def pick_or_ask(
    question: str,
    options: list[str],
    recommended: str,
    reason: str,
    user_level: str,
) -> str:
    """
    - no_tecnico: elige recommended automáticamente y lo explica.
    - tecnico: pregunta con menú.
    """
    if user_level == "no_tecnico":
        print("\n" + question)
        print(f"🤖 Voy a elegir por ti: {recommended}")
        print(f"🤖 Motivo: {reason}")
        return recommended

    return ask_menu(
        question,
        options,
        recommended=recommended,
        recommendation_reason=reason
    )


# ----------------------------
# Router con memoria (propuesta + explicación + pregunta)
# ----------------------------
def propose_type_with_explanation() -> dict:
    system = SystemMessage(
        content=(
            "Eres un asistente enfocado SOLO en configurar JSONs para modelos de ML.\n"
            "Si el usuario se sale del tema, debes volver a encauzar preguntando sobre el objetivo.\n"
            "Responde SOLO con un JSON válido (sin texto adicional).\n\n"
            "Clases:\n"
            '- "cluster": AGRUPAR cosas similares.\n'
            '- "series_tiempo": PREDECIR valores futuros en el tiempo.\n'
            '- "preguntar": si no está claro; devuelve UNA pregunta corta.\n\n'
            "Incluye una explicación MUY sencilla para que el usuario confirme."
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
def fill_cluster(user_level: str):
    cfg = dict(CLUSTER_SCHEMA)

    print("\n🧩 Configuración de CLUSTERING (agrupar elementos similares).")

    cfg["objetivo"] = pick_or_ask(
        "¿Qué quieres lograr?",
        CLUSTER_OPTIONS["objetivo"],
        recommended="segmentacion_clientes",
        reason="Es el caso más común para empezar: crear grupos con perfiles parecidos.",
        user_level=user_level
    )

    if cfg["objetivo"] == "deteccion_anomalias":
        rec_alg = "dbscan"
        rec_k = "auto"
        alg_reason = "Suele servir para detectar puntos raros sin fijar número de grupos."
        k_reason = "Para anomalías es mejor no forzar K."
    else:
        rec_alg = "kmeans"
        rec_k = "3"
        alg_reason = "Es simple y estable para empezar."
        k_reason = "3 es un punto de partida típico."

    cfg["algoritmo"] = pick_or_ask(
        "¿Qué algoritmo prefieres?",
        CLUSTER_OPTIONS["algoritmo"],
        recommended=rec_alg,
        reason=alg_reason,
        user_level=user_level
    )

    cfg["num_clusters"] = pick_or_ask(
        "¿Cuántos grupos quieres?",
        CLUSTER_OPTIONS["num_clusters"],
        recommended=rec_k,
        reason=k_reason,
        user_level=user_level
    )
    if cfg["num_clusters"] != "auto":
        cfg["num_clusters"] = int(cfg["num_clusters"])

    cfg["escalado"] = pick_or_ask(
        "¿Cómo quieres normalizar/escalar tus datos?",
        CLUSTER_OPTIONS["escalado"],
        recommended="zscore",
        reason="Ayuda cuando las variables están en escalas distintas.",
        user_level=user_level
    )

    cfg["metrica"] = pick_or_ask(
        "¿Qué métrica de distancia usar?",
        CLUSTER_OPTIONS["metrica"],
        recommended="euclidean",
        reason="La más común para datos numéricos.",
        user_level=user_level
    )

    return cfg


def fill_series(user_level: str):
    cfg = dict(SERIES_SCHEMA)

    print("\n📈 Configuración de SERIES DE TIEMPO (predecir futuro con historial).")

    cfg["frecuencia"] = pick_or_ask(
        "¿Con qué frecuencia están tus datos?",
        SERIES_OPTIONS["frecuencia"],
        recommended="diaria",
        reason="Es la frecuencia más común; si luego vemos que es semanal/mensual lo ajustamos.",
        user_level=user_level
    )

    cfg["horizonte"] = ask_int_in_range("¿Cuántos pasos hacia el futuro quieres predecir?", 1, 30)

    if cfg["frecuencia"] in ["diaria", "semanal"]:
        rec_modelo = "prophet"
        model_reason = "Es una opción amigable para empezar y maneja estacionalidad de forma sencilla."
    else:
        rec_modelo = "arima"
        model_reason = "Modelo clásico y efectivo para series más agregadas."

    cfg["modelo"] = pick_or_ask(
        "¿Qué modelo quieres usar?",
        SERIES_OPTIONS["modelo"],
        recommended=rec_modelo,
        reason=model_reason,
        user_level=user_level
    )

    cfg["metrica_error"] = pick_or_ask(
        "¿Cómo medimos qué tan bueno es el modelo?",
        SERIES_OPTIONS["metrica_error"],
        recommended="mae",
        reason="Es la más fácil de interpretar (error promedio en unidades reales).",
        user_level=user_level
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
# Chat principal (nivel -> entender -> confirmar -> llenar -> guardar)
# ----------------------------
print("💬 Asistente para crear configuración JSON de ML (escribe 'salir' para terminar)\n")

state = "nivel"
chosen_type = None
user_level = None

while True:
    if state == "nivel":
        print("👋 Antes de empezar, una pregunta rápida:")
        user_level = ask_menu(
            "¿Tienes conocimiento técnico en Machine Learning?",
            ["no_tecnico", "tecnico"],
            recommended="no_tecnico",
            recommendation_reason="Si no eres técnico, yo tomaré las decisiones por ti y te lo explicaré en sencillo."
        )
        state = "entender"
        continue

    if state == "entender":
        user_input = input("👤 Cuéntame tu problema (ej: 'segmentar clientes' o 'predecir ventas'): ").strip()
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 Hasta luego.")
            break

        # Guardar en memoria
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

                if is_out_of_scope(ans):
                    print("🤖 Solo puedo ayudarte a configurar el modelo/JSON. Responde la pregunta para avanzar.")
                    continue

                memory.chat_memory.add_user_message(ans)
                continue

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

            print("\n🤖 Perfecto. Dímelo en una frase (solo lo relacionado con el objetivo):")
            correction = input("👤 ¿Qué quieres lograr exactamente?: ").strip()
            if correction.lower() in ["salir", "exit", "quit"]:
                print("👋 Hasta luego.")
                raise SystemExit

            if is_out_of_scope(correction):
                print("🤖 Solo puedo ayudarte con la configuración del modelo. Por favor describe el objetivo.")
                continue

            memory.chat_memory.add_user_message(correction)

        if state != "llenar":
            chosen_type = ask_menu(
                "Para no bloquearte, elige lo que más se parezca a tu objetivo:",
                ["cluster", "series_tiempo"],
                recommended="series_tiempo"
            )
            state = "llenar"

    elif state == "llenar":
        if chosen_type == "cluster":
            cfg = fill_cluster(user_level)
        else:
            cfg = fill_series(user_level)

        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Listo. Generé el archivo interno: {RESULT_FILE}")
        print("\n🧠 Así se resolverá tu problema (en sencillo):")
        print("🤖 " + explain_solution(cfg) + "\n")
        break
