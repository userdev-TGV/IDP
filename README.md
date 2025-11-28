# Herramienta de Analisis de Contratos

Aplicacion para analizar contratos PDF con OCR y modelos de IA. El backend expone una API Flask y el front-end es React (Vite) con animaciones estilo InsightBot.

## Caracteristicas
- OCR en Azure Document Intelligence para extraer texto y paginas.
- Procesamiento en varios idiomas con prompts especificos y traduccion al ingles.
- Chat sobre el texto extraido.
- Descarga de resultados en JSON/CSV desde el front.

## Requisitos Previos
- Python 3.8+
- Node 18+
- Credenciales de Azure AI Document Intelligence (obligatorio para el OCR).
- Clave de OpenAI o Azure OpenAI.

## Instalacion
1) Clona el repo y entra en la carpeta:
```
git clone <url-del-repositorio>
cd IDP
```
2) Instala dependencias de Python:
```
pip install -r requirements.txt
```
3) Prepara las variables de entorno:
```
cp .env.example .env   # en Windows puedes copiar manualmente
```
Completa en `.env` tus claves: `AZURE_ENDPOINT`, `AZURE_KEY`, `OPENAI_API_KEY` o `AZURE_OPENAI_*`.
4) Instala dependencias del front:
```
cd frontend
npm install
cd ..
```

## Uso (Flask + React)
1) Levanta la API Flask (puerto 8000):
```
python api_server.py
```
2) En otra terminal arranca el front:
```
cd frontend
npm run dev -- --host
```
3) Abre el navegador en la URL que imprima Vite (por defecto http://localhost:5173). Si necesitas apuntar a otro backend crea `frontend/.env.local` con `VITE_API_BASE=http://host:puerto`.

## Endpoints principales
- `POST /api/analyze` (multipart form): campos `file`, `language` y opcional `customPrompt`.
- `POST /api/chat` (JSON): campos `question` y `extractedText` (la lista devuelta por `/api/analyze`).
- `GET /api/health`: estado del servicio.

## Solucion de problemas
- Verifica que `.env` tenga los valores de Azure y OpenAI.
- Si no se extrae texto, confirma que el PDF sea legible y que las credenciales de Azure sean correctas.
- Asegurate de no bloquear puertos 8000 (API) y 5173 (Vite).

## Licencia
MIT.
