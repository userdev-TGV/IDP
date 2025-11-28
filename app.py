import matplotlib.pyplot as plt
import pyodbc
import json
import time
import asyncio
import os
import re
import base64
import io
import requests
import pandas as pd
from rich.console import Console
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st

# Azure libraries
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# Add markitdown for Excel processing
try:
    import markitdown
except ImportError:
    pass  # Will handle the missing library case later


# Set page config as the first Streamlit command
st.set_page_config(
    page_title="IDP",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for modern UI (Spanish, Roboto, new colors)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif !important;
    }
    .main {
        background-color: #f7f7f7;
    }
    .stAppToolbar {visibility: hidden;}
    
    .stMainBlockContainer {
        padding: 2em !important;
    }
    
    .stVerticalBlockBorderWrapper{
        width: 35% !important;
        margin: 0 auto !important;
    }

    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        margin-top: 1em;
        background-color: #e30613;
        color: #fff;
        font-weight: bold;
        border: none;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color: #34374b;
        color: #fff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: #34374b;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 0.5rem 0 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        color: #fff;
        font-weight: 500;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e30613;
        color: #fff;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 0.5rem 0.5rem 0.5rem;
    }
    .stMarkdown, .stMetric, .stExpander, .uploadedFile {
        font-family: 'Roboto', sans-serif;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 6px solid #e30613;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 6px solid #34374b;
    }
    .uploadedFile {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 6px solid #4a4a49;
        box-shadow: 0 2px 8px rgba(52,55,75,0.05);
    }
    .stDownloadButton>button {
        background-color: #4a4a49;
        color: #fff;
        border-radius: 6px;
        font-weight: bold;
        border: none;
        margin-top: 0.5em;
    }
    .stDownloadButton>button:hover {
        background-color: #e30613;
        color: #fff;
    }
    .stAlert, .stInfo, .stWarning, .stError {
        border-radius: 8px;
        font-family: 'Roboto', sans-serif;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        border-radius: 6px;
        border: 1.5px solid #34374b;
        font-family: 'Roboto', sans-serif;
    }
    .stRadio>div>label {
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
    }
    .stSelectbox>div>div>div>div {
        font-family: 'Roboto', sans-serif;
    }
    .stFileUploader>div>div {
        border: 2px dashed #e30613;
        border-radius: 8px;
        background: #fff;
    }
    .stFileUploader>div>div:hover {
        border: 2px solid #34374b;
    }
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #e30613 0%, #34374b 100%);
    }
    .stChatMessage {
        background: #f0f2f6;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-family: 'Roboto', sans-serif;
    }
    .stContainer {
        font-family: 'Roboto', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables from .env file
load_dotenv()


# Azure credentials and configurations
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

# Azure OpenAI API key
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# LogicApps/PowerAutomate API URL for uploads to database
UPLOAD_URL = os.getenv("UPLOAD_URL")

# Create image directory if it doesn't exist
IMAGES_DIR = "processed_images"
os.makedirs(IMAGES_DIR, exist_ok=True)


# Initialize Azure client
document_client = DocumentAnalysisClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY)
)

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY)
else:
    # Use Azure OpenAI if API key is not provided
    openai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2025-01-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT)


from prompts import SYSTEM_PROMPTS, SYSTEM_PROMPT_ENGLISH

# ----------------- LOGGING UTILITY ----------------- #


class LogUtil:
    def __init__(self):
        self.info_console = Console(stderr=False, style="bold green")
        self.warn_console = Console(stderr=True, style="bold yellow")
        self.error_console = Console(stderr=True, style="bold red")

    def info(self, message):
        self.info_console.log(message)

    def warn(self, message):
        self.warn_console.log(message)

    def error(self, message):
        self.error_console.log(message)


# ----------------- CHART GENERATION LOGIC ----------------- #

class ChartUtil:
    def __init__(self):
        self.log = LogUtil()
        self.client = openai_client
        self.chart_out_path = os.path.join("assets", "chart.png")

    def generate_chart(self, message):
        ci_prompt = (
            """You are a Python data analyst. Based on the following user data or description, 
    generate clean and valid Python code using matplotlib to produce a chart. 
    Do not include any markdown formatting (no ```), explanations, or comments. 
    Only return executable Python code. The code must define a variable called 'fig' containing the plot.\n\n
    """+message
        )

        try:
            if OPENAI_API_KEY:
                thread = self.client.beta.threads.create(
                    messages=[{"role": "user", "content": ci_prompt}]
                )

                run = self.client.beta.threads.runs.create(
                    assistant_id=OPENAI_ASSISTANT_ID,
                    thread_id=thread.id
                )

                while True:
                    run = self.client.beta.threads.runs.retrieve(
                        run_id=run.id, thread_id=thread.id
                    )

                    if run.status == "completed":
                        self.log.info("✅ Chart generation completed.")

                        messages = self.client.beta.threads.messages.list(
                            thread_id=thread.id)
                        self.log.info(messages.data[0])

                        image_file_id = messages.data[0].content[0].image_file.file_id
                        content_description = messages.data[0].content[1].text.value

                        raw_response = self.client.files.with_raw_response.content(
                            file_id=image_file_id)
                        self.client.files.delete(image_file_id)

                        with open(self.chart_out_path, "wb") as f:
                            f.write(raw_response.content)
                            return (self.chart_out_path, content_description)

                    elif run.status == "failed":
                        self.log.error("❌ Chart generation failed.")
                        break

                    time.sleep(1)
            else:
                st.info("Using Azure OpenAI API for chart generation.")
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst who only response with matplotlib code"},
                        {"role": "user", "content": ci_prompt}
                    ]
                )

                code = response.choices[0].message.content
                self.log.info(code)

                local_vars = {}
                exec(code, {"plt": plt}, local_vars)

                fig = local_vars.get("fig")
                if fig:
                    fig.savefig(self.chart_out_path)
                    return (self.chart_out_path, "Chart generated successfully.")
                else:
                    self.log.error(
                        "❌ No 'fig' object found in the generated code.")
        except Exception as e:
            self.log.error(
                f"An unexpected error occurred during chart generation process: {e}")

        return (None, "🤔 Could you please rephrase your query and try again?")


# ----------------- UI CLASS FOR CHARTS ----------------- #
class ChartUI:
    def __init__(self):
        self.chart_util = ChartUtil()
        self.log = LogUtil()

        st.markdown("""
            <style>
                .stButton > button {
                    background-color: white;
                    color: black; ;
                    font-weight: bold;
                    border-radius: 8px;
                    padding: 0.5em 1.5em;
                    transition: background-color 0.3s ease;
                }
                .reportview-container .main .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                }
                footer {visibility: hidden;}
                h2, h3 {
                    color: #333333;
                }
                .chart-desc {
                    font-style: italic;
                    color: #666;
                    padding-top: 0.5rem;
                }
            </style>
        """, unsafe_allow_html=True)

    def generate_and_render(self, user_input: str):
        if not user_input.strip():
            st.warning("⚠️ Por favor, ingresa algunos datos o una descripción.")
            return

        with st.spinner("⏳ Generando gráfica..."):
            image_path, description = self.chart_util.generate_chart(
                user_input)

        with st.container(border=False, height=450):
            if image_path:
                # Usamos columnas para centrar la imagen
                col1, col2, col3 = st.columns([1, 2, 1])  # proporciones
                with col2:
                    st.image(image_path, use_container_width=True)
            else:
                st.error("❌ No se pudo generar la gráfica. Por favor, intenta de nuevo.")
                # ----------------- STREAMLIT TAB RENDER FUNCTION ----------------- #
def render_chart_tab():
    st.markdown("## 📊 Generador de Gráficas")
    st.markdown(
        "Utiliza esta herramienta para **generar gráficas** a partir de datos estructurados o descripciones en texto libre usando IA.")

    input_text = st.text_area(
        "✏️ Ingresa tus datos o una descripción de la gráfica que deseas", height=100)

    if st.button("📊 Generar Gráfica", key=1):
        chart_ui = ChartUI()
        chart_ui.generate_and_render(input_text)


def create_modal(content):
    """Create a loading modal with the given content."""
    modal_style = """
    <style>
        .modal {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            position: fixed;
            z-index: 1;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: #fefefe;
            margin: auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            max-width: 500px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """
    modal_html = f"""
    <div class="modal">
        <div class="modal-content">
            {content}
            <p>Processing your contract document...</p>
        </div>
    </div>
    """
    return modal_style + modal_html


def preprocess_string(s):
    """Normalize string for comparison."""
    if isinstance(s, (int, float)):
        return str(s)
    s = str(s)
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = ' '.join(s.split())
    return s.lower()


def flexible_string_match(s1, s2):
    """Compare strings with preprocessing."""
    return preprocess_string(s1) == preprocess_string(s2)


def compare_dates(date1, date2):
    """Compare dates in different formats."""
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y']
    for fmt in formats:
        try:
            d1 = datetime.strptime(date1, fmt)
            d2 = datetime.strptime(date2, fmt)
            return d1 == d2
        except ValueError:
            continue
    return False


async def process_document(file_bytes, file_type, file_name, custom_prompt=None, language="English"):
    """Process document using Textract and OpenAI."""
    try:
        textract_start_time = time.time()
        all_extracted_text = []
        image_paths = []

        if file_type == "application/pdf":
            status_text = st.empty()
            status_text.text("Processing PDF with Azure Form Recognizer...")
            poller = document_client.begin_analyze_document(
                model_id="prebuilt-read", document=file_bytes)
            result = poller.result()

            for page_num, page in enumerate(result.pages, start=1):
                for line in page.lines:
                    all_extracted_text.append({
                        'text': line.content,
                        'confidence': 0.98,
                        'page': page_num
                    })
        else:
            # Use Azure Form Recognizer on the image
            poller = document_client.begin_analyze_document(
                model_id="prebuilt-read", document=file_bytes)
            result = poller.result()

            # Extract text from this page
            page_text = []
            for page in result.pages:
                for line in page.lines:
                    all_extracted_text.append({
                        'text': line.content,
                        'confidence': 0.97,
                        'page': 1
                    })

        textract_end_time = time.time()
        textract_duration = textract_end_time - textract_start_time

        if not all_extracted_text:
            st.error("No text was extracted from the document.")
            return None, None, None

        status_text = st.empty()
        status_text.text("Processing with OpenAI...")

        # Process with OpenAI using the selected language prompt
        openai_start_time = time.time()
        if custom_prompt:
            # Use custom prompt if provided
            system_prompt = custom_prompt
        else:
            # Use selected language prompt
            system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_ENGLISH)

        openai_response = process_with_openai(
            all_extracted_text,
            system_prompt
        )
        openai_end_time = time.time()
        openai_duration = openai_end_time - openai_start_time

        # Calculate metrics
        metrics = {
            "textract_duration": textract_duration,
            "openai_duration": openai_duration,
            "total_text_lines": len(all_extracted_text),
            "page_count": len(image_paths) if image_paths else 1,
            "saved_images": image_paths
        }

        return openai_response, metrics, all_extracted_text

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, None, None


def process_with_openai(text, system_prompt):
    """Process OCR text with OpenAI to extract structured data."""
    try:
        # Format text with page numbers if available
        formatted_text = "\n".join([
            f"[Page {item.get('page', 1)}] {item['text']} (Confidence: {item['confidence']:.2f})"
            for item in text
        ])

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    "Process the following Portuguese contract text following "
                    "the system instructions and return the data in JSON format. "
                    f"Each line is followed by its confidence score:\n\n{formatted_text}"
                )}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error processing with OpenAI: {str(e)}")
        return {"error": "Failed to process with OpenAI"}


def chat_with_document(extracted_text, question):
    """Chat with the extracted document content using OpenAI."""
    try:
        # Format text with page numbers if available
        formatted_text = "\n".join([
            f"[Page {item.get('page', 1)}] {item['text']}"
            for item in extracted_text
        ])

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that answers questions about contract "
                    "documents. Use only the information provided in the document "
                    "text. If the answer is not in the document, say you don't know."
                )},
                {"role": "user", "content": (
                    "Here is the content of a contract document:\n\n"
                    f"{formatted_text}\n\n"
                    f"Answer this question about the document: {question}"
                )}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error chatting with OpenAI: {str(e)}")
        return "Error: Failed to process with OpenAI"


def compare_json_structures(openai_json, comparison_json):
    """Compare expected and extracted JSON structures."""
    if isinstance(openai_json, str):
        try:
            openai_json = json.loads(openai_json)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in OpenAI results"}

    if isinstance(comparison_json, str):
        try:
            comparison_json = json.loads(comparison_json)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in comparison data"}

    match_count = 0
    total_count = 0
    mismatches = []

    def compare_recursive(openai_part, comparison_part, path=""):
        nonlocal match_count, total_count, mismatches

        if isinstance(openai_part, dict) and isinstance(comparison_part, dict):
            for key in set(openai_part.keys()) | set(comparison_part.keys()):
                new_path = f"{path}.{key}" if path else key
                if key in openai_part and key in comparison_part:
                    compare_recursive(
                        openai_part[key],
                        comparison_part[key],
                        new_path
                    )
                else:
                    total_count += 1
                    mismatches.append(f"Missing key: {new_path}")
        elif isinstance(openai_part, list) and isinstance(comparison_part, list):
            for i, (openai_item, comparison_item) in enumerate(
                zip(openai_part, comparison_part)
            ):
                new_path = f"{path}[{i}]"
                compare_recursive(openai_item, comparison_item, new_path)
        else:
            total_count += 1
            if flexible_string_match(openai_part, comparison_part):
                match_count += 1
            elif compare_dates(str(openai_part), str(comparison_part)):
                match_count += 1
            else:
                mismatches.append(
                    f"Mismatch at {path}: OpenAI: {openai_part}, "
                    f"Comparison: {comparison_part}"
                )

    compare_recursive(openai_json, comparison_json)

    return {
        "match_count": match_count,
        "total_count": total_count,
        "match_percentage": (match_count / total_count * 100) if total_count > 0 else 0,
        "mismatches": mismatches
    }


def add_banner(png_file):
    """Add a PNG banner to the Streamlit app."""
    with open(png_file, "rb") as f:
        png = f.read()
    b64 = base64.b64encode(png).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" style="width:40%;height:auto;">'


def display_extraction_results(openai_results, metrics, extracted_text):
    """Display extraction results in a visually appealing way."""
    st.subheader("Extracted Contract Information")

    # Show page info if multi-page document
    if metrics.get("page_count", 1) > 1:
        st.info(f"Processed {metrics['page_count']} pages")

        # Display thumbnails of processed images if available
        if metrics.get("saved_images"):
            st.subheader("Processed Pages")
            image_cols = st.columns(min(4, len(metrics["saved_images"])))
            for i, image_path in enumerate(metrics["saved_images"]):
                col_idx = i % 4
                with image_cols[col_idx]:
                    st.image(image_path, caption=f"Page {i+1}", width=150)

        # === Main Contract Info ===
    contracts = openai_results.get("contracts", [])
    for idx, contract in enumerate(contracts):
        st.markdown(f"### 📄 Contract Payment Type {idx + 1}")

        col1, col2 = st.columns(2)
        simple_fields = [
            ("Contract", "Contract"),
            ("Contract Number", "Contract Number"),
            ("Contract Type", "Contract Type"),
            ("Customer", "Customer"),
            ("Region", "Region"),
            ("Effective Date", "Effective Date"),
            ("Expiration Date", "Expiration Date"),
            ("Contract Terms", "Contract Terms"),
            ("Payment Type", "Payment Type"),
            ("Payment Value", "Payment Value"),
            ("Currency", "Currency")
        ]

        for i, (key, label) in enumerate(simple_fields):
            col = col1 if i % 2 == 0 else col2
            value = contract.get(key, "Not found")
            col.markdown(f"**{label}:** {value}")

        # List-based fields
        list_fields = [
            ("Promo Tactic", "Promo Tactic(s)"),
            ("Product Category", "Product Category(ies)"),
            ("Incentives Details", "Incentives Details"),
            ("Legal Aspects", "Legal Aspects")
        ]
        for key, label in list_fields:
            items = contract.get(key, [])
            if isinstance(items, list) and items:
                st.markdown(f"**{label}:**")
                for item in items:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"**{label}:** Not found")

        # Multi-line text fields
        long_fields = [
            ("Promotion Display", "Promotion Display"),
            ("Payment Structure", "Payment Structure"),
            ("Penalties", "Penalties")
        ]
        for key, label in long_fields:
            st.markdown(f"**{label}:** {contract.get(key, 'Not found')}")

    # Show tabs for additional information
    tab1, tab2, tab3 = st.tabs(["Raw JSON", "OCR Text", "Processing Metrics"])

    with tab1:
        st.json(openai_results)

    with tab2:
        # Group by page if page info available
        page_grouped_text = {}
        for item in extracted_text:
            page = item.get('page', 1)
            if page not in page_grouped_text:
                page_grouped_text[page] = []
            page_grouped_text[page].append(item)

        if len(page_grouped_text) > 1:
            page_tabs = st.tabs(
                [f"Page {page}" for page in sorted(page_grouped_text.keys())])
            for i, page in enumerate(sorted(page_grouped_text.keys())):
                with page_tabs[i]:
                    st.json(page_grouped_text[page])
        else:
            st.json(extracted_text)

    with tab3:
        # Remove image paths from display
        display_metrics = {k: v for k,
                           v in metrics.items() if k != "saved_images"}
        st.json(display_metrics)

    # Add download buttons for this specific file
    col1, col2 = st.columns(2)
    file_name = "temp_results"  # Placeholder for file name

    with col1:
        json_str = json.dumps(openai_results, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{file_name}_results.json",
            mime="application/json",
            use_container_width=True
        )
    with col2:
        # Create a CSV for this single file
        csv_data = []
        for field, data in openai_results.items():
            if isinstance(data, dict) and "value" in data and "confidence" in data:
                csv_data.append({
                    "File": file_name,
                    "Field": field,
                    "Value": data["value"],
                    "Confidence": data["confidence"]
                })
        df = pd.DataFrame(csv_data)
        csv_string = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_string,
            file_name=f"{file_name}_results.csv",
            mime="text/csv",
            use_container_width=True
        )


def create_download_button(data, file_name, button_text):
    """Create data for a download button for a file."""
    json_str = json.dumps(data, indent=2)
    return json_str, file_name, button_text


def create_csv_download_button(all_results, file_name):
    """Create a download button for CSV containing multiple extraction results."""
    # Prepare data for CSV
    csv_data = []
    for file_result in all_results:
        doc_name = file_result.get("file_name", "unknown")
        extraction_data = file_result.get("extraction_data", {})

        for field, data in extraction_data.items():
            if isinstance(data, dict) and "value" in data and "confidence" in data:
                csv_data.append({
                    "File": doc_name,
                    "Field": field,
                    "Value": data["value"],
                    "Confidence": data["confidence"]
                })

    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(csv_data)
    csv_string = df.to_csv(index=False)

    # Return the CSV string and filename to be used with st.download_button
    return csv_string, file_name


def create_modern_results_display(openai_results, metrics, extracted_text):
    """Display extraction results in a modern, visually appealing way."""
    # Create metric cards for key information
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Contract Number",
            value=openai_results.get("contract_number", {}).get(
                "value", "Not found"),
            delta=f"Confidence: {openai_results.get('contract_number', {}).get('confidence', 0):.2%}"
        )

    with col2:
        st.metric(
            label="Customer",
            value=openai_results.get("customer", {}).get("value", "Not found"),
            delta=f"Confidence: {openai_results.get('customer', {}).get('confidence', 0):.2%}"
        )

    # Create expandable sections for detailed information
    with st.expander("📋 Detailed Contract Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            for field in ["region", "effective_date", "expiration_date"]:
                if field in openai_results:
                    st.markdown(f"""
                        <div class="stMetric">
                            <h4>{field.replace('_', ' ').title()}</h4>
                            <p>{openai_results[field]['value']}</p>
                            <p>Confidence: {openai_results[field]['confidence']:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)

        with col2:
            for field in ["contract_term", "promo_tactic", "category"]:
                if field in openai_results:
                    st.markdown(f"""
                        <div class="stMetric">
                            <h4>{field.replace('_', ' ').title()}</h4>
                            <p>{openai_results[field]['value']}</p>
                            <p>Confidence: {openai_results[field]['confidence']:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)

    # Create tabs for raw data
    tab1, tab2, tab3 = st.tabs(["📊 Raw JSON", "📝 OCR Text", "📈 Metrics"])

    with tab1:
        st.json(openai_results)

    with tab2:
        if len(extracted_text) > 1:
            page_tabs = st.tabs(
                [f"Page {i+1}" for i in range(len(extracted_text))])
            for i, tab in enumerate(page_tabs):
                with tab:
                    st.json(extracted_text[i])
        else:
            st.json(extracted_text)

    with tab3:
        display_metrics = {k: v for k,
                           v in metrics.items() if k != "saved_images"}
        st.json(display_metrics)


def show_chat_interface(extracted_text):
    """Muestra la interfaz de chat con el texto extraído."""
    st.subheader("Haz preguntas sobre tu contrato")

    # Inicializa el historial de chat si no existe
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Muestra el historial de chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Entrada de chat
    user_question = st.chat_input("Escribe una pregunta sobre tu contrato")

    if user_question:
        # Agrega el mensaje del usuario al historial
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Muestra el mensaje del usuario
        with st.chat_message("user"):
            st.write(user_question)

        # Obtiene la respuesta de la IA
        with st.spinner("Pensando..."):
            ai_response = chat_with_document(extracted_text, user_question)

        # Agrega la respuesta de la IA al historial
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response
        })

        # Muestra la respuesta de la IA
        with st.chat_message("assistant"):
            st.write(ai_response)

        # Botón para limpiar historial
        if st.button("🧹 Limpiar historial de chat"):
            st.session_state.chat_history = []
            st.rerun()


def chat_tab():
    """Interfaz para chatear con el documento de contrato."""
    st.markdown("""
    En esta pestaña puedes hacer preguntas sobre el contrato que has subido.
    Puedes usar un documento previamente procesado o subir uno nuevo. El sistema puede analizar contratos en varios idiomas.
    """)

    # Placeholders para la interfaz
    modal_placeholder = st.empty()
    chat_placeholder = st.empty()

    # Verifica si hay resultados previos
    has_previous_extraction = (
        'extracted_text' in st.session_state and
        st.session_state.extracted_text is not None
    )

    # Botón de selección de documento
    if has_previous_extraction:
        document_source = st.radio(
            "📄 Selecciona el origen del documento:",
            ["Usar documento previamente procesado", "Subir nuevo documento"],
            horizontal=True
        )
    else:
        document_source = "Subir nuevo documento"

    if document_source == "Usar documento previamente procesado":
        st.success("Usando documento previamente procesado para el chat")
        show_chat_interface(st.session_state.extracted_text)
    else:
        # Subida de archivo para chat
        uploaded_file = st.file_uploader(
            "📁 Selecciona un documento de contrato",
            type=["pdf", "png", "jpg", "jpeg"],
            key="chat_file_uploader"
        )

        if uploaded_file is not None:
            # Muestra información del archivo
            file_type = uploaded_file.type
            file_name = os.path.splitext(uploaded_file.name)[0]
            st.info(f"Archivo subido: {uploaded_file.name} ({file_type})")

            if st.button("🚀 Procesar para chat", use_container_width=True):
                try:
                    # Muestra modal de procesamiento
                    logo_html = add_banner("assets/icon.png")
                    modal_html = create_modal(logo_html)
                    modal_placeholder.markdown(
                        modal_html, unsafe_allow_html=True)
                    extracted_text = []

                    # Procesa el documento
                    with st.spinner("Procesando documento para chat..."):
                        file_bytes = uploaded_file.getvalue()

                        if file_type == "application/pdf":
                            status_text = st.empty()
                            status_text.text(
                                "Procesando PDF con Azure Form Recognizer...")
                            poller = document_client.begin_analyze_document(
                                model_id="prebuilt-read", document=file_bytes)
                            result = poller.result()

                            for page_num, page in enumerate(result.pages, start=1):
                                for line in page.lines:
                                    extracted_text.append({
                                        'text': line.content,
                                        'confidence': 0.98,
                                        'page': page_num
                                    })
                        else:
                            # Usa Azure Form Recognizer en la imagen
                            poller = document_client.begin_analyze_document(
                                model_id="prebuilt-read", document=file_bytes)
                            result = poller.result()

                            for page in result.pages:
                                for line in page.lines:
                                    extracted_text.append({
                                        'text': line.content,
                                        'confidence': 0.98,
                                        'page': 1
                                    })

                        # Guarda en session state
                        st.session_state.extracted_text = extracted_text

                        # Limpia historial de chat para nuevo documento
                        if 'chat_history' in st.session_state:
                            st.session_state.chat_history = []

                        # Limpia el modal
                        modal_placeholder.empty()

                        # Mensaje de éxito y muestra interfaz de chat
                        st.success(
                            "¡Documento procesado! Ahora puedes chatear con él.")
                        show_chat_interface(extracted_text)

                except Exception as e:
                    modal_placeholder.empty()
                    st.error(f"Error procesando el documento: {str(e)}")


def chat_with_database(data, question):
    """Chat with the database content using OpenAI."""
    try:
        # Convert dataframe to a formatted string representation
        data_str = data.to_string(index=False)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that answers questions about contract "
                    "databases. Use only the information provided in the Excel "
                    "data. If the answer is not in the data, say you don't know. "
                    "When appropriate, refer to specific rows or entries from the data."
                )},
                {"role": "user", "content": (
                    "Here is the content of a contract database:\n\n"
                    f"{data_str}\n\n"
                    f"Answer this question about the database: {question}"
                )}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error chatting with OpenAI: {str(e)}")
        return "Error: Failed to process with OpenAI"


def show_database_chat_interface(data):
    """Muestra la interfaz de chat para la base de datos con los datos dados."""
    st.subheader("Haz preguntas sobre tu base de datos de contratos")

    # Inicializa historial si no existe
    if 'db_chat_history' not in st.session_state:
        st.session_state.db_chat_history = []

    # Muestra historial
    for message in st.session_state.db_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Entrada de chat
    user_question = st.chat_input(
        "Escribe una pregunta sobre tu base de datos de contratos")

    if user_question:
        # Agrega mensaje del usuario
        st.session_state.db_chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Muestra mensaje del usuario
        with st.chat_message("user"):
            st.write(user_question)

        # Obtiene respuesta de la IA
        with st.spinner("Pensando..."):
            ai_response = chat_with_database(data, user_question)

        # Agrega respuesta de la IA
        st.session_state.db_chat_history.append({
            "role": "assistant",
            "content": ai_response
        })

        # Muestra respuesta de la IA
        with st.chat_message("assistant"):
            st.write(ai_response)


def process_excel_with_markitdown(excel_data):
    """Process Excel data with markitdown to enhance its structure."""
    try:
        # Check if markitdown is available
        if 'markitdown' not in globals():
            st.warning(
                "Markitdown library not available. Using standard pandas processing.")
            return excel_data

        # Convert DataFrame to a temporary Excel file
        temp_file = "temp_excel.xlsx"
        excel_data.to_excel(temp_file, index=False)

        # Process with markitdown
        md_result = markitdown.excel_to_markdown(temp_file)

        # Parse the resulting markdown structure back into a DataFrame
        # This is a simplified approach - actual implementation depends on markitdown's output
        structured_data = pd.DataFrame()
        try:
            # Parse the markdown to extract structured data
            headers = []
            rows = []

            lines = md_result.split('\n')
            for i, line in enumerate(lines):
                if i == 0:  # Headers
                    headers = [h.strip() for h in line.split('|') if h.strip()]
                elif i > 1:  # Skip the separator line
                    if '|' in line:
                        row_data = [cell.strip()
                                    for cell in line.split('|') if cell]
                        if row_data:
                            rows.append(row_data)

            if headers and rows:
                structured_data = pd.DataFrame(rows, columns=headers)
            else:
                structured_data = excel_data  # Fallback
        except Exception:
            st.warning("Error parsing markitdown output. Using original data.")
            structured_data = excel_data

        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return structured_data
    except Exception as e:
        return excel_data  # Return original data if processing fails


def load_sql_database(query="SELECT * FROM Contracts"):
    """Connect to Azure SQL and return a DataFrame."""
    conn_str = os.getenv("AZURE_SQL_CONNECTION_STRING")
    if not conn_str:
        st.error("Azure SQL connection string not found in environment variables.")
        return None
    try:
        with pyodbc.connect(conn_str) as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error connecting to Azure SQL: {e}")
        return None


def chat_database_tab():
    """Interfaz para chatear con la base de datos de contratos en Azure SQL."""
    st.markdown("""
    En esta pestaña puedes chatear con una base de datos de contratos almacenada en Azure SQL.
    """)
    # Muestra chat si los datos ya están cargados
    if st.session_state.get("database_loaded", False):
        show_database_chat_interface(st.session_state["excel_data"])
        if st.button("📊 Generar Gráfica", key=2):
            text = st.session_state["db_chat_history"][-1]
            chart_ui = ChartUI()
            chart_ui.generate_and_render(text.get("content", ""))
        if st.button("🔄 Reiniciar y cargar otra base de datos"):
            st.session_state.pop("excel_data", None)
            st.session_state.pop("db_chat_history", None)
            st.session_state["database_loaded"] = False
            st.rerun()
        if st.button("🧹 Limpiar historial de chat"):
            st.session_state.db_chat_history = []
            st.rerun()
        return

    # Entrada de consulta SQL (opcional, o usa la predeterminada)
    query = st.text_area(
        "Consulta SQL para cargar datos de contratos:",
        value="SELECT * FROM Contracts",
        help="Edita la consulta SQL si deseas filtrar o unir tablas.")
    if st.button("🔗 Cargar base de datos desde Azure SQL", type="primary", use_container_width=True):
        with st.spinner("Cargando datos desde Azure SQL..."):
            df = load_sql_database(query)
            if df is not None and not df.empty:
                st.session_state.excel_data = df
                st.session_state.database_loaded = True
                st.success(f"Se cargaron {len(df)} registros desde Azure SQL.")
                st.subheader("Vista previa de la base de datos")
                st.dataframe(df.head(5))
                show_database_chat_interface(df)
            else:
                st.error("No se cargaron datos desde Azure SQL.")


def upload_to_database(data):
    # Send POST request to API
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        UPLOAD_URL, json=data, headers=headers)

    if response.status_code in [200, 202, 203]:
        st.success(
            f"Uploaded records to the database successfully!")
    else:
        st.error(
            f"Failed to upload data to the database. Status code: {response.status_code}")


def main():
    """Main function to run the Streamlit app."""
    # Header empresarial
    st.markdown("""
            <div style='display: flex; align-items: center; gap: 2rem; background: #34374b; padding: 1.5rem 2rem; border-radius: 0 0 16px 16px; box-shadow: 0 2px 8px rgba(52,55,75,0.08); margin-bottom: 2rem;'> 
            <div style='flex:1; display: flex; flex-direction: column; align-items: center; justify-content: center;'>
                <h1 style='color: #fff; font-size: 2.2rem; margin-bottom: 0.2rem; font-family: Roboto, sans-serif; text-align: center;'>Extracción y Análisis de Contratos Multilingüe</h1>
                <p style='color: #e30613; font-size: 1.1rem; margin: 0; font-family: Roboto, sans-serif; text-align: center;'>Extrae, analiza y consulta tus contratos en una interfaz moderna, intuitiva y empresarial.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs with icons
    tab0, tab1, tab3, tab4, tab5= st.tabs([
        "🏠 Descripción de la herramienta",
        "📑 Extracción Estándar",
        "💬 Chat con Contrato",
        "🗄️ Chat con Base de Datos",
        "📊 Generar Gráficas"
    ])

    with tab0:
        # Beneficios y descripción (antes en sidebar)
        st.markdown("""
        <div style='background: #f7f7f7; border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(52,55,75,0.04);'>
            <h3 style='color: #34374b; margin-top: 0; font-family: Roboto, sans-serif;'>¿Qué puedes hacer con esta herramienta?</h3>
            <ul style='color: #4a4a49; font-size: 1.08rem; line-height: 1.7; font-family: Roboto, sans-serif;'>
                <li>Extrae información clave de contratos en varios idiomas (inglés, español, ruso, portugués)</li>
                <li>Procesa múltiples archivos a la vez</li>
                <li>Chatea con tus documentos de contrato</li>
                <li>Chatea con la base de datos de contratos</li>
                <li>Plantillas de extracción personalizadas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    # Tab 1: Standard extraction
    with tab1:
        st.markdown("""
            ### Extracción Estándar de Información de Contratos
            Sube tus contratos para extraer automáticamente información clave en el idioma que elijas.
            
            Puedes cargar varios archivos y procesarlos todos al mismo tiempo.
        """)

        # Add language selector
        selected_language = st.selectbox(
            "🌐 Selecciona el idioma para la extracción:",
            options=list(SYSTEM_PROMPTS.keys()),
            index=0  # Por defecto Inglés
        )

        # File uploader with modern styling
        uploaded_files = st.file_uploader(
            "📁 Arrastra y suelta tus archivos aquí o haz clic para seleccionarlos",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="standard_file_uploader"
        )

        if uploaded_files:
            # Display uploaded files in a grid
            cols = st.columns(2)
            for i, file in enumerate(uploaded_files):
                with cols[i % 2]:
                    st.markdown(f"""
                        <div class=\"uploadedFile\">
                            <p><strong>{file.name}</strong></p>
                            <p>Tamaño: {file.size / 1024:.1f} KB</p>
                            <p>Tipo: {file.type}</p>
                        </div>
                    """, unsafe_allow_html=True)

            if st.button("🚀 Procesar todos los documentos", use_container_width=True):
                # Initialize list to store all results
                all_results = []

                # Process files and show results
                with st.spinner("Procesando todos los documentos..."):
                    for file_index, file in enumerate(uploaded_files):
                        try:
                            st.markdown(
                                f"### Procesando archivo {file_index + 1}/{len(uploaded_files)}: {file.name}")
                            progress_bar = st.progress(0)

                            file_bytes = file.getvalue()
                            file_type = file.type
                            file_name = os.path.splitext(file.name)[0]

                            # Update progress
                            progress_bar.progress(25)

                            openai_results, metrics, extracted_text = asyncio.run(
                                process_document(
                                    file_bytes, file_type, file_name, None, selected_language)
                            )

                            # Update progress
                            progress_bar.progress(90)

                            if openai_results and metrics:
                                # Store results in session state
                                result_data = {
                                    "file_name": file.name,
                                    "extraction_data": openai_results,
                                    "metrics": metrics,
                                    "extracted_text": extracted_text,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }

                                # Add to all results list
                                all_results.append(result_data)

                                # Update processed files list
                                if 'processed_files' not in st.session_state:
                                    st.session_state.processed_files = []

                                file_already_processed = any(
                                    processed['file_name'] == file.name
                                    for processed in st.session_state.processed_files
                                )

                                if file_already_processed:
                                    st.session_state.processed_files = [
                                        result_data if processed['file_name'] == file.name
                                        else processed
                                        for processed in st.session_state.processed_files
                                    ]
                                else:
                                    st.session_state.processed_files.append(
                                        result_data)

                                # Store the most recent in session state for chat tab
                                st.session_state.extracted_text = extracted_text
                                st.session_state.latest_results = openai_results

                                # Complete progress
                                progress_bar.progress(100)

                                upload_to_database(openai_results)

                                # Create an expander for each file's results
                                with st.expander(f"Resultados para {file.name}", expanded=False):
                                    # Display results
                                    # Instead of calling display_extraction_results which uses expanders,
                                    # we'll show the results directly to avoid nested expanders
                                    st.subheader(
                                        "Información extraída del contrato")

                                    # Show page info if multi-page document
                                    if metrics.get("page_count", 1) > 1:
                                        st.info(
                                            f"Se procesaron {metrics['page_count']} páginas")

                                        # Display thumbnails of processed images if available
                                        if metrics.get("saved_images"):
                                            st.subheader("Páginas procesadas")
                                            image_cols = st.columns(
                                                min(4, len(metrics["saved_images"])))
                                            for i, image_path in enumerate(metrics["saved_images"]):
                                                col_idx = i % 4
                                                with image_cols[col_idx]:
                                                    st.image(
                                                        image_path, caption=f"Page {i+1}", width=150)

                                    # Create two columns for main results
                                    col1, col2 = st.columns(2)

                                    # === Main Contract Info ===
                                    contracts = openai_results.get(
                                        "contracts", [])
                                    for idx, contract in enumerate(contracts):
                                        st.markdown(
                                            f"### 📄 Tipo de pago del contrato {idx + 1}")

                                        col1, col2 = st.columns(2)
                                        simple_fields = [
                                            ("Contract", "Contrato"),
                                            ("Contract Number", "Número de contrato"),
                                            ("Contract Type", "Tipo de contrato"),
                                            ("Customer", "Cliente"),
                                            ("Region", "Región"),
                                            ("Effective Date", "Fecha de vigencia"),
                                            ("Expiration Date", "Fecha de expiración"),
                                            ("Contract Terms", "Términos del contrato"),
                                            ("Payment Type", "Tipo de pago"),
                                            ("Payment Value", "Valor del pago"),
                                            ("Currency", "Moneda")
                                        ]

                                        for i, (key, label) in enumerate(simple_fields):
                                            col = col1 if i % 2 == 0 else col2
                                            value = contract.get(
                                                key, "No encontrado")
                                            col.markdown(
                                                f"**{label}:** {value}")

                                        # List-based fields
                                        list_fields = [
                                            ("Promo Tactic", "Táctica promocional(es)"),
                                            ("Product Category", "Categoría(s) de producto"),
                                            ("Incentives Details", "Detalles de incentivos"),
                                            ("Legal Aspects", "Aspectos legales")
                                        ]
                                        for key, label in list_fields:
                                            items = contract.get(key, [])
                                            if isinstance(items, list) and items:
                                                st.markdown(f"**{label}:**")
                                                for item in items:
                                                    st.markdown(f"- {item}")
                                            else:
                                                st.markdown(
                                                    f"**{label}:** No encontrado")

                                        # Multi-line text fields
                                        long_fields = [
                                            ("Promotion Display", "Visualización de la promoción"),
                                            ("Payment Structure", "Estructura de pago"),
                                            ("Penalties", "Penalizaciones")
                                        ]
                                        for key, label in long_fields:
                                            st.markdown(
                                                f"**{label}:** {contract.get(key, 'No encontrado')}")

                                    # Show tabs for additional information
                                    tab1, tab2, tab3 = st.tabs([
                                        "🗂️ JSON Extraído", "📝 Texto OCR", "📊 Métricas de Procesamiento"])

                                    with tab1:
                                        st.json(openai_results)

                                    with tab2:
                                        # Group by page if page info available
                                        page_grouped_text = {}
                                        for item in extracted_text:
                                            page = item.get('page', 1)
                                            if page not in page_grouped_text:
                                                page_grouped_text[page] = []
                                            page_grouped_text[page].append(
                                                item)

                                        if len(page_grouped_text) > 1:
                                            page_tabs = st.tabs(
                                                [f"Page {page}" for page in sorted(page_grouped_text.keys())])
                                            for i, page in enumerate(sorted(page_grouped_text.keys())):
                                                with page_tabs[i]:
                                                    st.json(
                                                        page_grouped_text[page])
                                        else:
                                            st.json(extracted_text)

                                    with tab3:
                                        st.markdown("**Métricas del procesamiento:**")
                                        st.write(metrics)

                            else:
                                st.error(
                                    f"Processing failed for {file.name}. Skipping to next file.")
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")

                # If we have processed multiple files, show a summary and bulk download options
                if len(all_results) > 1:
                    st.markdown("### Resumen de Procesamiento por Lote")
                    st.success(
                        f"Se han procesado exitosamente {len(all_results)} de {len(uploaded_files)} documento(s)!"
                    )

                    # Add bulk download options
                    st.markdown("### Descargar Todos los Resultados")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Combined JSON download
                        combined_json = {
                            "batch_results": [
                                {
                                    "file_name": result["file_name"],
                                    "extraction_data": result["extraction_data"],
                                    "timestamp": result["timestamp"]
                                }
                                for result in all_results
                            ],
                            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        json_str = json.dumps(combined_json, indent=2)
                        st.download_button(
                            label="📥 Descargar Todos los Resultados (JSON)",
                            data=json_str,
                            file_name="batch_results.json",
                            mime="application/json",
                            use_container_width=True
                        )

                    with col2:
                        # Combined CSV download
                        csv_string, csv_filename = create_csv_download_button(
                            all_results, "batch_results.csv")
                        st.download_button(
                            label="📥 Descargar Todos los Resultados (CSV)",
                            data=csv_string,
                            file_name=csv_filename,
                            mime="text/csv",
                            use_container_width=True
                        )

                elif len(all_results) == 1:
                    st.success(f"Se ha procesado exitosamente el documento!")

                # Add comparative analysis option if multiple documents
                if len(all_results) > 1:
                    st.markdown("### Análisis Comparativo")
                    with st.expander("Comparar Campos en Documentos"):
                        # Create a comparison dataframe
                        comparison_data = []
                        for result in all_results:
                            file_name = result["file_name"]
                            extraction_data = result["extraction_data"]

                            for field, data in extraction_data.items():
                                if isinstance(data, dict) and "value" in data and "confidence" in data:
                                    comparison_data.append({
                                        "File": file_name,
                                        "Field": field,
                                        "Value": data["value"],
                                        "Confidence": data["confidence"]
                                    })

                        # Convert to DataFrame and pivot for comparison
                        if comparison_data:
                            df = pd.DataFrame(comparison_data)
                            pivot_df = df.pivot_table(
                                index="Field",
                                columns="File",
                                values=["Value", "Confidence"],
                                aggfunc="first"
                            )

                            # Display the comparison table
                            st.dataframe(pivot_df)
    # Tab3: Chat with Contract
    with tab3:
        st.markdown("""
            ### 💬 Chat con Contrato
            Haz preguntas en lenguaje natural sobre el contenido del contrato procesado y recibe respuestas inteligentes.
        """)
        chat_tab()  # Asume que la función ya maneja la lógica, solo traducimos la interfaz

    # Tab 4: Chat with database
    with tab4:
        st.markdown("""
            ### 🗄️ Chat con Base de Datos de Contratos
            Conéctate a la base de datos de contratos y realiza preguntas inteligentes sobre tus datos almacenados.
        """)
        chat_database_tab()  # Asume que la función ya maneja la lógica, solo traducimos la interfaz

    # Tab 5: Generate charts
    with tab5:
        render_chart_tab()  # Asume que la función ya maneja la lógica, solo traducimos la interfaz


if __name__ == "__main__":
    main()
