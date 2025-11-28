# Herramienta de Análisis de Contratos

Aplicación para analizar contratos PDF con OCR de Azure AI Document Intelligence y modelos de OpenAI/Azure OpenAI. El backend es una API FastAPI y el front-end es un cliente React moderno inspirado en InsightBot.

## Funciones principales
1. **Extracción de Datos Estándar**: Extrae información clave del contrato (número, cliente, fechas, tácticas promocionales, etc.).
2. **Extracción Personalizada**: Permite definir tus propias instrucciones de extracción para contratos.
3. **Preguntas y Respuestas**: Chat sobre el contenido extraído de cada documento.

## Características
- **Soporte multilingüe**: procesamiento en varios idiomas y normalización en inglés.
- **Procesamiento de múltiples archivos**: carga simultánea y resultados organizados por documento.
- **OCR en la nube**: Azure AI Document Intelligence (prebuilt-read) para extraer texto y metadatos.
- **Análisis con IA**: prompts optimizados para contratos con OpenAI/Azure OpenAI.
- **Interfaz moderna**: flujo de carga, progreso, resultados y chat con animaciones en React/Vite.

## Requisitos previos
- Python 3.8+ (se recomienda 3.10+)
- Node.js 18+ (para el cliente React)
- Credenciales de Azure AI Document Intelligence (obligatorio)
- Clave de OpenAI **o** credenciales de Azure OpenAI (obligatorio)
- Opcionales: credenciales de Azure SQL y SharePoint/MSAL según tus integraciones

## Configuración de variables de entorno
1. Copia el archivo `.env.example` a `.env` en la raíz del proyecto.
2. Sustituye los placeholders por tus credenciales reales. Nunca subas el archivo `.env` al control de versiones.
3. Variables principales:
   - `AZURE_ENDPOINT` y `AZURE_KEY`: credenciales de Azure AI Document Intelligence.
   - `OPENAI_API_KEY` **o** `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`: acceso al modelo de chat.
   - `AZURE_SQL_CONNECTION_STRING` y `AZURE_SQL_AUTH_CONNECTION_STRING`: cadenas de conexión opcionales para Azure SQL.
   - `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`: opcionales para descargas desde SharePoint.
   - `VITE_API_BASE`: URL del backend cuando el front-end React no consume `http://localhost:8000`.

## Instalación y ejecución
### Backend (FastAPI)
1. Instala dependencias Python:
   ```bash
   pip install -r requirements.txt
   ```
2. Levanta la API (requiere `.env` configurado):
   ```bash
   uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
   ```

### Front-end (React/Vite)
1. En otra terminal:
   ```bash
   cd frontend
   npm install
   npm run dev -- --host
   ```
2. Abre el navegador en la URL indicada por Vite (por defecto http://localhost:5173). Define `VITE_API_BASE` en `frontend/.env.local` si necesitas apuntar a otro backend.

## Campos de extracción de contratos
La herramienta extrae los siguientes campos (traducidos al inglés): número de contrato, cliente, región, fecha de vigencia, fecha de expiración, duración del contrato, táctica promocional, categoría, pago/descuento y moneda.

## Solución de problemas
- **Credenciales faltantes**: la API requiere `AZURE_ENDPOINT` y `AZURE_KEY` configurados.
- **Errores de modelo**: valida tu clave de OpenAI o Azure OpenAI y tu cuota disponible.
- **CORS o URL del backend**: ajusta `VITE_API_BASE` en el front-end si sirves la API en otra ruta.
- **SharePoint**: si habilitas descargas, completa los valores de MSAL (`AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`).

## Licencia
Proyecto bajo licencia MIT. Consulta `LICENSE` para más detalles.
