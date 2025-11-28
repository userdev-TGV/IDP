# Herramienta de Análisis de Contratos

Una aplicación para analizar contratos PDF utilizando OCR e IA. Ahora puedes elegir entre la experiencia clásica en Streamlit o un nuevo front-end React con animaciones estilo InsightBot.

### Funciones principales
1. **Extracción de Datos Estándar**: Extrae información clave del contrato como número de contrato, cliente, fechas, etc.
2. **Extracción Personalizada**: Define tus propias reglas de extracción para contratos.
3. **Preguntas y Respuestas sobre Contratos**: Haz preguntas sobre tu contrato y obtén respuestas generadas por IA.

## Características

- **Soporte Multilingüe**:
  - Procesamiento de contratos en múltiples idiomas (Inglés, Español, Ruso, Portugués)
  - Traduce automáticamente la información extraída al inglés para mantener consistencia

- **Procesamiento de Múltiples Archivos**:
  - Carga y procesa varios archivos PDF simultáneamente
  - Interfaz de resultados organizada para cada documento
  - Análisis comparativo entre documentos

- **Opciones de OCR**:
  - AWS Textract para reconocimiento de texto de alta precisión
  - Los archivos PDF se convierten a imágenes para un mejor procesamiento
  - Extracción automática de texto, tablas y formularios

- **Análisis con IA**:
  - Extrae campos específicos de contratos utilizando la API de OpenAI
  - Responde preguntas sobre el contenido del contrato
  - Valida datos extraídos contra valores esperados

- **Opciones de Descarga**:
  - Descarga datos individuales de cada documento en formato JSON o CSV
  - Descarga combinada de todos los resultados procesados en batch
  - Exportación de análisis comparativo

- **Interfaz Amigable**:
  - Carga fácil de archivos
  - Presentación de datos en formato tabular y visual
  - Indicadores de progreso durante el procesamiento
  - Interfaz de chat interactiva

## Requisitos Previos

- Python 3.8+ instalado
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) instalado (para la opción de OCR local)
- Poppler instalado (para procesamiento de PDF con pdf2image)
- Clave API de OpenAI (para análisis con IA)
- Credenciales de [Azure AI Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) (obligatorio para realizar OCR en documentos PDF) 

## Instalación

1. Clona este repositorio:
   ```
   git clone <url-del-repositorio>
   cd herramienta-analisis-contratos
   ```

2. Instala los paquetes Python requeridos:
   ```
   pip install -r requirements.txt
   ```

3. Instala Tesseract OCR y Poppler:
   - **Windows**: 
     - [Tesseract para Windows](https://github.com/UB-Mannheim/tesseract/wiki)
     - [Poppler para Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
   - **MacOS**:
     ```
     brew install tesseract poppler
     ```
   - **Linux**:
     ```
     sudo apt-get install tesseract-ocr poppler-utils
     ```

4. Configura las variables de entorno:
   - Copia el archivo `.env.example` a un nuevo archivo llamado `.env`
   - Llena tus claves API y credenciales en el archivo `.env`:
     ```
     # Clave API de OpenAI (Requerida para extracción y Q&A)
     OPENAI_API_KEY=your_openai_api_key_here

     # Para Azure Textract (obligatorio)
     AZURE_ENDPOINT=your_endpoint_link_here
     AZURE_KEY=your_api_key_here
     
     ```

## Uso

### Nuevo front-end React

1. Inicia la API FastAPI (requiere las variables de entorno configuradas):
   ```
   uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. En otra terminal instala y levanta el cliente React:
   ```
   cd frontend
   npm install
   npm run dev -- --host
   ```

3. Abre el navegador en la URL indicada por Vite (por defecto http://localhost:5173). Si necesitas apuntar a otro backend define `VITE_API_BASE` en un `.env.local` dentro de `frontend/`.

### Experiencia Streamlit

1. Inicia la aplicación Streamlit tradicional:
   ```
   streamlit run app.py
   ```

2. Abre tu navegador web en la URL mostrada en la terminal (normalmente http://localhost:8501)

3. Selecciona la pestaña que deseas utilizar:
   - **Extracción Estándar**: Para extraer información predefinida de contratos
   - **Extracción Personalizada**: Para definir tus propias reglas de extracción
   - **Chat con Documentos**: Para hacer preguntas sobre tu contrato

4. En la pestaña de Extracción Estándar:
   - Selecciona el idioma para el análisis (Inglés, Español, Ruso, Portugués)
   - Carga uno o múltiples archivos PDF de contratos
   - Haz clic en "Procesar Todos los Documentos"
   - Los resultados se mostrarán organizados por cada archivo
   - Puedes descargar resultados individuales en JSON o CSV
   - Si procesaste múltiples archivos, también puedes descargar todos los resultados combinados

5. En la pestaña de Extracción Personalizada:
   - Define tu propio prompt de extracción
   - Carga un contrato
   - Procesa el documento con tu prompt personalizado

6. En la pestaña de Chat:
   - Usa un documento previamente procesado o carga uno nuevo
   - Haz preguntas sobre el contenido del contrato
   - Recibe respuestas basadas en el texto extraído

## Campos de Extracción de Contratos

La herramienta extrae los siguientes campos de los contratos y los traduce al inglés:
- Número de contrato
- Cliente
- Región
- Fecha de vigencia
- Fecha de expiración
- Duración del contrato
- Táctica promocional
- Categoría
- Pago/Descuento
- Moneda

## Solución de Problemas

- **Problemas de OCR**: Si la extracción de texto es deficiente, verifica la calidad del PDF
- **Errores de API**: Verifica las claves API y asegúrate de tener suficientes créditos/cuota
- **Problemas de Instalación**: Asegúrate de que Tesseract y Poppler estén correctamente instalados y en tu PATH
- **Variables de Entorno**: Asegúrate de que tu archivo .env esté en el directorio raíz y tenga el formato correcto
- **Tiempos de Espera de AWS Textract**: Para documentos muy grandes, el trabajo asíncrono podría tardar más que el tiempo de espera predeterminado.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo LICENSE para más detalles.

## Agradecimientos

- [OpenAI](https://openai.com/) por las capacidades de IA
- [Streamlit](https://streamlit.io/) por el framework de aplicación web
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) para procesamiento OCR local
- [Azure AI Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) para análisis avanzado de documentos 