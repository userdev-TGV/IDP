import json
import os
import time
from datetime import datetime

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import AzureOpenAI, OpenAI

from prompts import SYSTEM_PROMPTS, SYSTEM_PROMPT_ENGLISH

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

if not AZURE_ENDPOINT or not AZURE_KEY:
    raise RuntimeError("Azure Form Recognizer credentials are required for the API server.")

if not (OPENAI_API_KEY or (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)):
    raise RuntimeError("OpenAI or Azure OpenAI credentials are required for the API server.")

app = Flask(__name__)
CORS(app)

document_client = DocumentAnalysisClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY),
)

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2025-01-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def process_document_bytes(file_bytes: bytes, file_type: str, file_name: str, custom_prompt: str | None, language: str):
    textract_start_time = time.time()
    all_extracted_text = []

    try:
        poller = document_client.begin_analyze_document(model_id="prebuilt-read", document=file_bytes)
        result = poller.result()
    except Exception as exc:
        raise RuntimeError(f"Azure Document Intelligence error: {exc}")

    page_count = len(result.pages)
    for page_num, page in enumerate(result.pages, start=1):
        for line in page.lines:
            all_extracted_text.append({
                "text": line.content,
                "confidence": 0.98,
                "page": page_num,
            })

    textract_duration = time.time() - textract_start_time
    if not all_extracted_text:
        return None, None, None

    system_prompt = custom_prompt or SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_ENGLISH)
    formatted_text = "\n".join(
        [f"[Page {item.get('page', 1)}] {item['text']} (Confidence: {item['confidence']:.2f})" for item in all_extracted_text]
    )

    openai_start_time = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Process the following contract text following the system instructions and return the data in JSON format. "
                    f"Each line is followed by its confidence score:\n\n{formatted_text}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    openai_duration = time.time() - openai_start_time
    metrics = {
        "textract_duration": textract_duration,
        "openai_duration": openai_duration,
        "total_text_lines": len(all_extracted_text),
        "page_count": page_count,
    }

    return json.loads(response.choices[0].message.content), metrics, all_extracted_text


def chat_with_document(extracted_text, question: str):
    formatted_text = "\n".join(
        [f"[Page {item.get('page', 1)}] {item['text']} (Confidence: {item.get('confidence', 1):.2f})" for item in extracted_text]
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for chatting about document extractions. Use only the provided text to answer.",
            },
            {"role": "user", "content": f"Document content:\n{formatted_text}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


@app.post("/api/analyze")
def analyze_document():
    file = request.files.get("file")
    language = request.form.get("language", "English")
    custom_prompt = request.form.get("customPrompt")

    if not file:
        return jsonify({"detail": "No file uploaded."}), 400

    file_bytes = file.read()
    try:
        openai_response, metrics, extracted_text = process_document_bytes(
            file_bytes, file.content_type, file.filename, custom_prompt, language
        )
    except Exception as exc:  # pragma: no cover - network dependency
        return jsonify({"detail": f"Processing error: {exc}"}), 500

    if not openai_response:
        return jsonify({"detail": "No text was extracted from the document."}), 400

    return jsonify({"openaiResults": openai_response, "metrics": metrics, "extractedText": extracted_text})


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")
    extracted_text = payload.get("extractedText")
    if not question or not extracted_text:
        return jsonify({"detail": "Both 'question' and 'extractedText' are required."}), 400
    answer = chat_with_document(extracted_text, question)
    return jsonify({"answer": answer})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
