from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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

# Inicializar FastAPI
app = FastAPI()

# Montar la carpeta de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Descargar recursos NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Cargar la base de datos desde el archivo
df = pd.read_csv('Emociones_BD.emociones.csv')

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

class ChatRequest(BaseModel):
    pregunta: str

# Ruta para servir la página HTML
@app.get("/", response_class=HTMLResponse)
async def get():
    with open("index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# Ruta para manejar las solicitudes de chat
@app.post("/api/chat")
async def chat(request: ChatRequest):
    pregunta = request.pregunta
    respuesta, _ = generar_respuesta_dialo(pregunta)
    return {"respuesta": respuesta}

# Ejecutar el servidor usando Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
