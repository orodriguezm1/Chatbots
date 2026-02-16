import os
import json
import warnings
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings("ignore")

# -----------------------
# 0) Setup
# -----------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ No se encontró OPENAI_API_KEY en el archivo .env")

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name="mistralai/mistral-7b-instruct",
    temperature=0.2,  # más estable para clasificación
)

memory = ConversationBufferMemory(return_messages=True)

# -----------------------
# 1) Funciones disponibles
# -----------------------
def suma(a, b):
    return a + b

def multiplica(a, b):
    return a * b

def suma_hasta(n):
    # 1 + 2 + ... + n
    if n < 1:
        return 0
    return n * (n + 1) // 2

def multiplica_hasta(n):
    # 1 * 2 * ... * n
    if n < 0:
        raise ValueError("n debe ser >= 0")
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res

FUNCTIONS = {
    "suma(a,b)": suma,
    "multiplica(a,b)": multiplica,
    "suma_hasta(n)": suma_hasta,
    "multiplica_hasta(n)": multiplica_hasta,
}

# -----------------------
# 2) Prompt del clasificador (agente básico)
# -----------------------
SYSTEM_CLASSIFIER = """
Eres un clasificador para un mini-agente.

Objetivo: en máximo 4 turnos (mensajes tuyos) identificar cuál de estas 4 funciones
debe ejecutarse. No inventes otras funciones.

Funciones EXACTAS posibles:
- suma(a,b)
- multiplica(a,b)
- suma_hasta(n)
- multiplica_hasta(n)

Reglas:
- Responde SIEMPRE en JSON válido y SOLO JSON.
- Si te falta información o la intención no está clara, responde:
  {"action":"ask","question":"<una sola pregunta corta para aclarar>"}
- Si ya estás seguro, responde:
  {"action":"choose","function":"<una de las 4 funciones EXACTAS>"}

Pistas:
- "sumar", "adicionar", "más" con dos números -> suma(a,b)
- "multiplicar", "por" con dos números -> multiplica(a,b)
- "sumar del 1 al n", "1+2+...+n" -> suma_hasta(n)
- "factorial", "producto del 1 al n", "1*2*...*n" -> multiplica_hasta(n)

Nunca pongas texto fuera del JSON.
"""

def extract_json(text: str) -> dict:
    """Extrae JSON aunque el modelo meta algo alrededor (por si acaso)."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    raise ValueError(f"No pude parsear JSON. Salida:\n{text}")

def call_classifier(user_msg: str) -> dict:
    """Llama al LLM usando memoria para clasificar / preguntar."""
    history = memory.load_memory_variables({}).get("history", [])
    messages = [SystemMessage(content=SYSTEM_CLASSIFIER)] + history + [HumanMessage(content=user_msg)]
    resp = llm.invoke(messages)
    data = extract_json(resp.content if hasattr(resp, "content") else str(resp))
    return data

# -----------------------
# 3) Recolección de parámetros + ejecución
# -----------------------
def ask_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ["salir", "exit", "quit"]:
            return None
        try:
            return int(raw)
        except ValueError:
            print("❌ Por favor ingresa un número entero (o 'salir').")

def ask_two_ints():
    a = ask_int("🔢 Ingresa a: ")
    if a is None:
        return None, None
    b = ask_int("🔢 Ingresa b: ")
    if b is None:
        return None, None
    return a, b

def execute_function(func_name: str):
    if func_name in ["suma(a,b)", "multiplica(a,b)"]:
        a, b = ask_two_ints()
        if a is None or b is None:
            return "EXIT"
        result = FUNCTIONS[func_name](a, b)
        print(f"✅ Resultado: {result}\n")
        return "OK"

    if func_name in ["suma_hasta(n)", "multiplica_hasta(n)"]:
        n = ask_int("🔢 Ingresa n: ")
        if n is None:
            return "EXIT"
        result = FUNCTIONS[func_name](n)
        print(f"✅ Resultado: {result}\n")
        return "OK"

    print("❌ Función desconocida (no debería pasar).")
    return "OK"

# -----------------------
# 4) Loop principal (agente por ciclos)
# -----------------------
print("💬 Mini-agente (máx 4 prompts para intención). Escribe 'salir' para terminar.\n")

while True:
    # Reinicio de ciclo
    classify_steps = 0
    chosen_function = None

    while chosen_function is None:
        user_input = input("👤 Tú: ").strip()
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 Hasta luego.")
            raise SystemExit

        # Guardar turno del usuario en memoria
        memory.chat_memory.add_message(HumanMessage(content=user_input))

        # Límite de 4 prompts al LLM para clasificar
        if classify_steps >= 4:
            print("🤖 Bot: No pude identificar la intención en 4 mensajes. Reformula tu solicitud.\n")
            # Puedes limpiar memoria del ciclo si quieres ser estricto:
            # memory.clear()
            break

        data = call_classifier(user_input)
        classify_steps += 1

        action = data.get("action")
        if action == "ask":
            question = (data.get("question") or "").strip()
            if not question:
                question = "¿Quieres sumar, multiplicar, sumar hasta n o multiplicar hasta n?"
            print(f"🤖 Bot: {question}\n")
            memory.chat_memory.add_message(SystemMessage(content=f"(bot_ask) {question}"))
            continue

        if action == "choose":
            fn = (data.get("function") or "").strip()
            if fn not in FUNCTIONS:
                # Si el modelo se sale de las 4 funciones, lo tratamos como ask
                print("🤖 Bot: ¿Puedes aclarar si quieres suma(a,b), multiplica(a,b), suma_hasta(n) o multiplica_hasta(n)?\n")
                continue

            # Regla: responder SOLO con el nombre exacto de la función
            print(fn)
            chosen_function = fn
            memory.chat_memory.add_message(SystemMessage(content=f"(bot_choose) {fn}"))
            break

        # Acción inválida -> pregunta genérica
        print("🤖 Bot: ¿Quieres suma(a,b), multiplica(a,b), suma_hasta(n) o multiplica_hasta(n)?\n")

    # Si no se eligió función (por límite), reinicia ciclo
    if chosen_function is None:
        continue

    # Pedir parámetros y ejecutar (fuera del LLM)
    status = execute_function(chosen_function)
    if status == "EXIT":
        print("👋 Hasta luego.")
        break

    # (Opcional) puedes limpiar memoria por ciclo si quieres “reinicio total”
    # memory.clear()
