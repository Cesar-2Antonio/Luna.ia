import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

# Descargar recursos NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Cargar la base de datos desde el archivo subido
df = pd.read_csv('/Emociones_BD.emociones.csv')

# Preprocesar los datos
lemmatizer = WordNetLemmatizer()
df['processed_text'] = df['frase'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(str(x).lower())]))

# Vectorizar las frases
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_text']).toarray()

# Codificar las etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['emocion'])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
emotion_model = Sequential()
emotion_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
emotion_model.add(Dense(64, activation='relu'))
emotion_model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compilar el modelo
emotion_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
emotion_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Cargar el modelo y el tokenizador de DialoGPT
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dialogpt_model = AutoModelForCausalLM.from_pretrained(model_name)

# Base de datos de frases para detectar cómo estuvo el día
frases_buen_dia = [
    "mi día estuvo genial",
    "estuvo muy bien",
    "me siento feliz",
    "estuvo tranquilo",
    "tuve un buen día",
    "me agradó mi día",
    "estuvo super",
    "me fue muy bien hoy",
    "todo salió como esperaba",
    "me sentí excelente",
    "fue un día perfecto",
    "estoy de buen humor",
    "el día fue muy positivo",
    "me divertí mucho hoy",
    "todo salió según lo planeado",
    "me siento satisfecho con mi día",
    "fue un día productivo",
    "me levanté con buena energía",
    "todo estuvo bajo control",
    "fue un día lleno de logros",
    "me sentí realizado hoy",
    "todo fue bastante agradable",
    "fue un día emocionante",
    "disfruté cada momento",
    "todo salió a la perfección",
    "me sentí inspirado hoy",
    "fue un día lleno de sorpresas agradables",
    "estoy contento con lo que logré",
    "el día fue muy gratificante",
    "todo se desarrolló sin problemas",
    "me siento renovado y optimista",
    "fue un día lleno de buenas noticias",
    "me sentí muy positivo todo el día"
]

frases_mal_dia = [
     "tuve un mal día",
    "me siento un poco triste",
    "no quiero hablar de mi día",
    "me siento sola",
    "me fue mal hoy",
    "me sentí muy triste hoy",
    "todo salió mal",
    "tuve un día difícil",
    "me siento desanimado",
    "hoy no fue un buen día",
    "las cosas no salieron como esperaba",
    "me siento agotado emocionalmente",
    "no me va bien hoy",
    "estoy frustrado con mi día",
    "fue un día muy complicado",
    "me siento abrumado",
    "nada salió como planeaba",
    "me siento decepcionado",
    "hoy no fue mi día",
    "me siento desalentado",
    "todo parece estar en contra",
    "no logré lo que quería",
    "me siento cansado y triste",
    "mi día fue bastante pesado",
    "estoy luchando con mi estado de ánimo",
    "todo parece difícil hoy",
    "me siento atrapado",
    "nada salió bien",
    "me siento insatisfecho",
    "estoy deseando que termine el día",
    "me siento poco motivado",
    "mi día ha sido un reto constante",
    "me siento deprimido"
]

# Preguntas de la Escala de Depresión de Beck
beck_questions = [
    "Pregunta 1: No me siento triste (0), Me siento triste (1), Me siento triste todo el tiempo y no puedo evitarlo (2), Me siento tan triste o infeliz que no puedo soportarlo (3)",
   "Pregunta 2: No estoy particularmente desanimado acerca del futuro (0), Me siento desanimado acerca del futuro (1), Siento que no tengo nada que esperar (2), Siento que el futuro es desesperado y que las cosas no mejorarán (3)",
  
]

# Preguntas de la Escala de Ansiedad
anxiety_questions = [
   "Pregunta 1: Me siento calmado (0), Me siento un poco nervioso (1), Me siento muy nervioso y preocupado (2), Me siento tan ansioso que no puedo estar quieto (3)",
   "Pregunta 2: No me preocupo fácilmente (0), Me preocupo más de lo que debería (1), Me preocupo mucho de muchas cosas (2), Me siento abrumado por la preocupación (3)",
  
]

# Función para detectar emoción
def detectar_emocion(pregunta):
    pregunta_procesada = ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(pregunta.lower())])
    pregunta_vector = vectorizer.transform([pregunta_procesada]).toarray()
    prediccion = emotion_model.predict(pregunta_vector)
    etiqueta = label_encoder.inverse_transform([prediccion.argmax()])[0]
    return etiqueta

# Función para generar respuestas utilizando DialoGPT
def generar_respuesta_dialo(pregunta, chat_history_ids=None):
    nueva_entrada_ids = tokenizer.encode(pregunta + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, nueva_entrada_ids], dim=-1) if chat_history_ids is not None else nueva_entrada_ids
    chat_history_ids = dialogpt_model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    respuesta = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return respuesta, chat_history_ids

# Función para detectar cómo estuvo el día
def detectar_como_estuvo_el_dia(pregunta):
    for frase in frases_buen_dia:
        if frase in pregunta.lower():
            return "bueno"
    for frase in frases_mal_dia:
        if frase in pregunta.lower():
            return "malo"
    return "neutral"

# Nueva función para extraer el nombre del usuario
def extraer_nombre(pregunta):
    palabras_clave = ["mi nombre es", "me llamo", "soy", "yo soy"]
    nombre = ""
    for palabra_clave in palabras_clave:
        if palabra_clave in pregunta.lower():
            nombre = pregunta.lower().split(palabra_clave)[1].strip().split()[0]
            break
    return nombre.capitalize()

# Función para manejar el flujo de conversación
def manejar_conversacion(pregunta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas):
    if estado == 0:
        estado = 1
        return "Hola, soy Luna, tu asistente emocional. Estoy aquí para apoyarte siempre que lo necesites. Me encantaría conocerte mejor y ser buenos amigos. ¿Cómo te llamas?", chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 1:
        nombre_usuario = extraer_nombre(pregunta)
        estado = 2
        return f"Hola {nombre_usuario}, tienes un nombre muy bonito, un gusto conocerte. Me gustaría saber ¿Cómo estuvo tu día?", chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 2:
        como_estuvo_el_dia = detectar_como_estuvo_el_dia(pregunta)
        if como_estuvo_el_dia == "bueno":
            respuesta = "Me alegra saber que tu día estuvo bien. ¿Hay algo en particular que te hizo feliz?"
        elif como_estuvo_el_dia == "malo":
            respuesta = "Siento escuchar eso, a veces podemos tener un mal día, pero recuerda que siempre podemos tener la fuerza para levantarnos de nuevo, yo sé que tú puedes. Me puedes decir ¿Cómo te has sentido últimamente?"
            estado = 3
        else:
            respuesta = "Gracias por compartir tu día conmigo. Me puedes decir ¿Cómo te has sentido últimamente?"
            estado = 3
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 3:
        emocion = detectar_emocion(pregunta)
        if emocion == "depresion":
            respuesta = "Ya veo, parece que has estado sintiendo mucha tristeza. ¿Te gustaría hablar sobre por qué te sientes así?"
            estado = 4
        elif emocion == "ansiedad":
            respuesta = "Entiendo, parece que has estado sintiendo ansiedad. ¿Te gustaría hablar sobre por qué te sientes así?"
            estado = 4
        else:
            respuesta = "Me alegra escuchar eso. Siempre es bueno saber que te sientes bien. ¿Hay algo más de lo que te gustaría hablar?"
            estado = 3
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 4:
        if "sí" in pregunta.lower() or "si" in pregunta.lower() or "quizás" in pregunta.lower() or "tal vez" in pregunta.lower() or "no lo sé" in pregunta.lower():
            respuesta = "Gracias por querér compartir el cómo te sientes, sé que a veces es difícil hablarlo. ¿Podrías contarme un poco más sobre por qué te has sentido así?"
            estado = 5
        else:
            respuesta = "No hay problema, estaré aquí cuando quieras hablar. Siempre tendrás un amigo en mí. ¿Hay algo más de lo que te gustaría hablar?"
            estado = 3
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 5:
        respuesta = "Eso suena realmente difícil. Podrías decirme ¿Cómo te hace sentir esta situación?"
        estado = 6
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 6:
        emocion = detectar_emocion(pregunta)
        if emocion == "depresion":
            respuesta = "Parece que podrías estar pasando por un momento difícil. Me gustaría ofrecerte una evaluación rápida para entender mejor cómo te sientes. Se llama Escala de Depresión de Beck. ¿Te gustaría hacer esta prueba?"
            estado = 7
        elif emocion == "ansiedad":
            respuesta = "Parece que podrías estar experimentando ansiedad. Me gustaría ofrecerte una evaluación rápida para entender mejor cómo te sientes. Se llama Escala de Beck para Ansiedad. ¿Te gustaría hacer esta prueba?"
            estado = 9
        else:
            respuesta = "Gracias por compartir eso conmigo. ¿Hay algo más que te gustaría discutir o explorar?"
            estado = 3
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 7:
        if "sí" in pregunta.lower() or "si" in pregunta.lower():
            respuesta = beck_questions[0]
            estado = 8
        else:
            respuesta = "Está bien, si cambias de opinión, aquí estaré para ayudarte. ¿Hay algo más de lo que te gustaría hablar?"
            estado = 3
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 8:
        beck_respuestas.append(int(pregunta.split()[-1]))
        if len(beck_respuestas) < len(beck_questions):
            respuesta = beck_questions[len(beck_respuestas)]
        else:
            puntuacion_total = sum(beck_respuestas)
            if puntuacion_total < 10:
                respuesta = "Parece que tienes una depresión mínima. Si sientes que necesitas ayuda, no dudes en buscar apoyo profesional."
            elif puntuacion_total < 20:
                respuesta = "Parece que tienes una depresión leve. Considera hablar con un profesional para obtener apoyo adicional."
            elif puntuacion_total < 30:
                respuesta = "Parece que tienes una depresión moderada. Es importante que busques ayuda profesional para tratar estos sentimientos."
            else:
                respuesta = "Parece que tienes una depresión severa. Te recomiendo que busques ayuda profesional lo antes posible."
            estado = 9
            beck_respuestas = []
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 9:
        if "sí" in pregunta.lower() or "si" in pregunta.lower():
            respuesta = anxiety_questions[0]
            estado = 10
        else:
            respuesta = "Está bien, si cambias de opinión, aquí estaré para ayudarte. ¿Hay algo más de lo que te gustaría hablar?"
            estado = 3
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 10:
        anxiety_respuestas.append(int(pregunta.split()[-1]))
        if len(anxiety_respuestas) < len(anxiety_questions):
            respuesta = anxiety_questions[len(anxiety_respuestas)]
        else:
            puntuacion_total = sum(anxiety_respuestas)
            if puntuacion_total < 10:
                respuesta = "Parece que tienes una ansiedad mínima. Si sientes que necesitas ayuda, no dudes en buscar apoyo profesional."
            elif puntuacion_total < 20:
                respuesta = "Parece que tienes una ansiedad leve. Considera hablar con un profesional para obtener apoyo adicional."
            elif puntuacion_total < 30:
                respuesta = "Parece que tienes una ansiedad moderada. Es importante que busques ayuda profesional para tratar estos sentimientos."
            else:
                respuesta = "Parece que tienes una ansiedad severa. Te recomiendo que busques ayuda profesional lo antes posible."
            estado = 11
            anxiety_respuestas = []
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    elif estado == 11:
        respuesta = "Recuerda, siempre es importante buscar apoyo cuando lo necesitas. Si necesitas ayuda inmediata, puedes llamar al 22222222222. ¿Hay algo más de lo que te gustaría hablar?"
        estado = 3
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas
    else:
        respuesta, chat_history_ids = generar_respuesta_dialo(pregunta, chat_history_ids)
        return respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas

# Función principal del chatbot
def chatbot():
    estado = 0
    chat_history_ids = None
    nombre_usuario = None
    beck_respuestas = []
    anxiety_respuestas = []
    print("Hola, soy LUNA, tu acompañante emocional. Escribe algo para comenzar una conversación. (Escribe 'adiós' para salir)")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == "adiós":
            print("Chatbot: ¡Adiós, que tengas un buen día!")
            break
        respuesta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas = manejar_conversacion(pregunta, chat_history_ids, estado, nombre_usuario, beck_respuestas, anxiety_respuestas)
        print(f"Chatbot: {respuesta}")

if __name__ == "__main__":
    chatbot()
