# Usar una imagen base ligera de Python 3.10
FROM python:3.10-slim

# Evitar que Python escriba archivos .pyc en el disco y habilitar logs instantáneos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema operativo (C++ build tools y librerías matemáticas)
# Necesario para compilar librerías complejas como scikit-learn, TA, o TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar solo el archivo de requerimientos primero (Aprovechar la caché de Docker)
COPY requirements.txt /app/

# Instalar las librerías de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente al contenedor
COPY . /app/

# Crear la carpeta de resultados por si no existe en el contenedor
RUN mkdir -p /app/results/saved_models /app/data/processed

# Comando por defecto para arrancar el bot de trading (Inferencia)
CMD ["python", "src/execution/main_bot.py"]
