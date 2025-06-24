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
    page_title="Contract Analysis Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        margin-top: 1em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
    }
    .stMarkdown {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .uploadedFile {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
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


# System prompts for different languages
SYSTEM_PROMPT_ENGLISH = """
You will be provided with one or more contract documents. Your task is to extract and summarize key contract information in a structured JSON format.
If a single contract includes more than one Payment Type (e.g., different discount types), generate one JSON object per Payment Type, duplicating the rest of the contract information accordingly.

Output Format:
Return a JSON object containing a list of contract entries, one per contract identified across the documents. Use the following structure:

{
  "contracts": [
    {
      "Contract": "Contract name with country in parentheses (e.g., 'Colombia (Inversiones Ríos Gallego)')",
      "Contract Number": "Unique contract identification number",
      "Contract Type": "Type of contract (e.g., 'Development Plan Agreement')",
      "Customer": "Full legal name of the customer",
      "Region": "Geographic area covered by the contract",
      "Effective Date": YYYY-MM-DD,
      "Expiration Date": YYYY-MM-DD,
      "Contract Terms": "Summary of key terms related to duration, renewals, or terminations",
      "Promo Tactic": ["List of primary promotional strategies/tactics used"],
      "Product Category": ["List of product categories covered"],
      "Payment Type": "Name or description of payment type or discount",
      "Payment Value": "Percentage value or amount from payment type",
      "Currency": "Currency specified in the contract",
      "Incentives Details": [
        "Bullet point list of all incentives, rebates, and discounts",
        "Include any conditional requirements"
      ],
      "Promotion Display": "Description of display/visibility agreements (e.g., shelf space, endcap display)",
      "Payment Structure": "Explanation of when and how payments/incentives are applied (e.g., quarterly, after target achieved)",
      "Penalties": "List or description of penalties for non-compliance or breach",
      "Legal Aspects": [
        "List of legal considerations (e.g., anti-corruption clauses, data protection, exclusivity terms)"
      ]
    }
  ]
}
 
"""

SYSTEM_PROMPT_SPANISH = """
Analiza el/los documento(s) proporcionado(s) y extrae la siguiente información en formato estructurado (JSON). Si se proporcionan múltiples documentos, crea una fila por cada contrato identificado.
Unicamente responde con el JSON manteniendo el formato del ejemplo de salida con la informacion extraida

| Campo                  | Descripción                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Contrato**           | Nombre del contrato/país (ej: "Colombia (Inversiones Ríos Gallego)")        |
| **Contract Number**    | Número de identificación del contrato                                       |
| **Contract Type**      | Tipo de contrato (ej: "Convenio Plan de Desarrollo")                        |
| **Customer**           | Nombre legal del cliente                                                    |
| **Region**             | Ámbito geográfico cubierto                                                  |
| **Effective Date**     | Fecha de inicio de vigencia                                                 |
| **Expiration Date**    | Fecha de terminación                                                        |
| **Contract Terms**     | Términos clave de duración y renovación                                     |
| **Promo Tactic**       | Estrategias promocionales principales                                       |
| **Product Category**   | Categorías de productos incluidos                                           |
| **Payment %**          | Porcentajes de descuento/incentivos (detallar tipos)                        |
| **Currency**           | Moneda del contrato                                                         |
| **Incentives Details** | Detalles específicos de incentivos/descuentos                               |
| **Promotion Display**  | Acuerdos sobre exhibición en puntos de venta                                |
| **Payment Structure**  | Método y plazos de aplicación de incentivos                                |
| **Penalties**          | Sanciones por incumplimiento                                               |
| **Legal Aspects**      | Aspectos legales relevantes (anticorrupción, protección de datos, etc.)    |

Instrucciones específicas:
1. Para campos con múltiples valores (como Payment %), usar viñetas o listas claras
2. Mantener la información precisa y fiel al documento original
3. Incluir detalles relevantes de anexos cuando corresponda
4. Para fechas, usar formato DD.MM.YYYY
5. Destacar valores porcentuales con el símbolo % correspondiente
6. En documentos en otros idiomas, traducir los términos clave al español pero mantener los nombres propios en original

Ejemplo de salida esperada (basado en documento colombiano):
| Contrato | Colombia (Inversiones Ríos Gallego) |
| Contract Number | 4072587 |
| Contract Type | Convenio Plan de Desarrollo 2019 |
| Customer | Inversiones Ríos Gallego S.A.S. |
| Region | Medellín y puntos de venta nacionales |
| Effective Date | 01.05.2019 |
| Expiration Date | 30.04.2020 |
| Contract Terms | 12 meses, renovable |
| Promo Tactic | Descuentos por desempeño operativo (exhibición, manejo de información) |
| Product Category | Huggies, Scott, Kleenex, Kotex, Poise, Plenitud |
| Payment % | [ Exhibición: 9%, No devoluciones: 0.3%, Manejo de Info: 1.2%, Eventos: 7%] |
| Currency | COP |
| Incentives Details | [ Exhibición: 9% por mamut en Medellín, Manejo de Info: 1.2% por reporte mensual, Eventos: 7% por participación en actividades comerciales] |
| Promotion Display | Exhibición "adicional": mamut/lateral en puntos estratégicos (San Antonio, Manrique) |
| Payment Structure | Notas de crédito (≤30 días). Base: facturación neta de devoluciones |
| Penalties | Multas por corrupción o incumplimiento de políticas |
| Legal Aspects | Cláusulas anticorrupción y protección de datos (Ley 1581/2012) |
"""

SYSTEM_PROMPT_RUSSIAN = """
Анализируйте предоставленный(ые) документ(ы) и извлеките следующую информацию в структурированном формате (JSON). Если предоставлено несколько документов, создайте строку для каждого выявленного контракта.
Отвечайте только JSON-данными, сохраняя формат примера вывода с извлеченной информацией.
| Поле                  | Описание                                                               |
|----------------------|------------------------------------------------------------------------|
| **Контракт**        | Название контракта/страна (например, "Колумбия (Inversiones Ríos Gallego)") |
| **Номер контракта** | Идентификационный номер контракта                                      |
| **Тип контракта**   | Тип контракта (например, "Соглашение о плане развития")               |
| **Клиент**         | Юридическое название клиента                                           |
| **Регион**         | Географический охват                                                  |
| **Дата вступления в силу** | Дата начала действия контракта                           |
| **Дата окончания** | Дата завершения контракта                                             |
| **Условия контракта** | Ключевые условия, касающиеся срока действия и продления          |
| **Промо-стратегия** | Основные рекламные стратегии                                       |
| **Категория продукции** | Включенные категории продукции                              |
| **Процент оплаты**  | Процент скидок/поощрений (указать типы)                          |
| **Валюта**        | Валюта контракта                                                     |
| **Детали поощрений** | Конкретные детали по скидкам и поощрениям                      |
| **Размещение рекламы** | Соглашения о размещении продукции в торговых точках         |
| **Структура выплат** | Метод и сроки применения поощрений                           |
| **Штрафы**         | Штрафы за несоблюдение условий контракта                        |
| **Юридические аспекты** | Важные юридические аспекты (антикоррупция, защита данных и др.) |

Специфические инструкции:
1. Для полей с несколькими значениями (например, "Процент оплаты") используйте маркированные списки.
2. Убедитесь, что информация точна и соответствует оригинальному документу.
3. Включайте важные детали из приложений, если это необходимо.
4. Используйте формат **ДД.ММ.ГГГГ** для дат.
5. Отмечайте процентные значения соответствующим символом **%**.
6. В документах на других языках переводите ключевые термины на **испанский**, но сохраняйте собственные имена в оригинале.

Ожидаемый формат вывода (на основе документа из Колумбии):
| Контракт | Колумбия (Inversiones Ríos Gallego) |
| Номер контракта | 4072587 |
| Тип контракта | Соглашение о плане развития 2019 |
| Клиент | Inversiones Ríos Gallego S.A.S. |
| Регион | Медельин и торговые точки по всей стране |
| Дата вступления в силу | 01.05.2019 |
| Дата окончания | 30.04.2020 |
| Условия контракта | 12 месяцев, продлеваемый |
| Промо-стратегия | Скидки за операционные показатели (размещение, управление информацией) |
| Категория продукции | Huggies, Scott, Kleenex, Kotex, Poise, Plenitud |
| Процент оплаты | [ Размещение: 9%, Отсутствие возвратов: 0.3%, Управление информацией: 1.2%, События: 7%] |
| Валюта | COP |
| Детали поощрений | [ Размещение: 9% за "мамут" в Медельине, Управление информацией: 1.2% за ежемесячный отчет, События: 7% за участие в коммерческих мероприятиях] |
| Размещение рекламы | "Дополнительное" размещение: "мамут"/боковая панель в стратегических местах (Сан-Антонио, Манрике) |
| Структура выплат | Кредитные ноты (≤30 дней). База: чистая сумма счета после возвратов |
| Штрафы | Штрафы за коррупцию или нарушение политик |
| Юридические аспекты | Антикоррупционные и законы о защите данных (Закон 1581/2012) |
"""

SYSTEM_PROMPT_PORTUGUESE = """
Analise o(s) documento(s) fornecido(s) e extraia as seguintes informações em formato estruturado (JSON). Se vários documentos forem fornecidos, crie uma linha para cada contrato identificado.
Responda apenas com o JSON, mantendo o formato do exemplo de saída com as informações extraídas.
| Campo                 | Descrição                                                               |
|----------------------|------------------------------------------------------------------------|
| **Contrato**        | Nome do contrato/país (ex: "Colômbia (Inversiones Ríos Gallego)")      |
| **Número do Contrato** | Número de identificação do contrato                                |
| **Tipo de Contrato**   | Tipo de contrato (ex: "Acordo do Plano de Desenvolvimento")       |
| **Cliente**        | Nome legal do cliente                                                |
| **Região**         | Abrangência geográfica                                             |
| **Data de Início** | Data de início da vigência                                       |
| **Data de Expiração** | Data de término do contrato                                   |
| **Termos do Contrato** | Condições-chave sobre duração e renovação                  |
| **Tática Promocional** | Principais estratégias promocionais                       |
| **Categoria de Produto** | Categorias de produtos incluídas                      |
| **Percentual de Pagamento** | Percentuais de desconto/incentivos (detalhar tipos) |
| **Moeda**        | Moeda do contrato                                                |
| **Detalhes de Incentivos** | Detalhes específicos sobre incentivos/descontos     |
| **Exibição Promocional** | Acordos sobre a exibição nos pontos de venda         |
| **Estrutura de Pagamento** | Método e prazos de aplicação dos incentivos       |
| **Penalidades**      | Penalidades por descumprimento                          |
| **Aspectos Legais**  | Aspectos jurídicos relevantes (anticorrupção, proteção de dados, etc.) |

Instruções específicas:
1. Para campos com múltiplos valores (como Percentual de Pagamento), use marcadores ou listas claras.
2. Mantenha a informação precisa e fiel ao documento original.
3. Inclua detalhes relevantes dos anexos quando aplicável.
4. Use o formato **DD.MM.AAAA** para datas.
5. Destaque valores percentuais com o símbolo **%** correspondente.
6. Em documentos em outros idiomas, traduza os termos-chave para **espanhol**, mas mantenha os nomes próprios no idioma original.

Exemplo de saída esperada (baseado em um documento colombiano):
| Contrato | Colômbia (Inversiones Ríos Gallego) |
| Número do Contrato | 4072587 |
| Tipo de Contrato | Acordo do Plano de Desenvolvimento 2019 |
| Cliente | Inversiones Ríos Gallego S.A.S. |
| Região | Medellín e pontos de venda nacionais |
| Data de Início | 01.05.2019 |
| Data de Expiração | 30.04.2020 |
| Termos do Contrato | 12 meses, renovável |
| Tática Promocional | Descontos por desempenho operacional (exibição, gestão de informações) |
| Categoria de Produto | Huggies, Scott, Kleenex, Kotex, Poise, Plenitud |
| Percentual de Pagamento | [ Exibição: 9%, Sem devoluções: 0.3%, Gestão de Informações: 1.2%, Eventos: 7%] |
| Moeda | COP |
| Detalhes de Incentivos | [Exibição: 9% para "mamut" em Medellín, Gestão de Informações: 1.2% por relatório mensal, Eventos: 7% por participação em atividades comerciais] |
| Exibição Promocional | Exibição "adicional": "mamut"/painel lateral em locais estratégicos (San Antonio, Manrique) |
| Estrutura de Pagamento | Notas de crédito (≤30 dias). Base: faturamento líquido após devoluções |
| Penalidades | Multas por corrupção ou descumprimento de políticas |
| Aspectos Legais | Cláusulas anticorrupção e proteção de dados (Lei 1581/2012) |
"""

# Map the prompts to their language names
SYSTEM_PROMPTS = {
    "English": SYSTEM_PROMPT_ENGLISH,
    "Spanish": SYSTEM_PROMPT_SPANISH,
    "Russian": SYSTEM_PROMPT_RUSSIAN,
    "Portuguese": SYSTEM_PROMPT_PORTUGUESE
}

# Define the default prompt as English
SYSTEM_PROMPT = SYSTEM_PROMPT_ENGLISH

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
                st.info("Using OpenAI API for chart generation.")
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
                    font-family: 'Segoe UI', sans-serif;
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
            st.warning("⚠️ Please enter some data or description.")
            return

        with st.spinner("⏳ Generating chart..."):
            image_path, description = self.chart_util.generate_chart(
                user_input)

        with st.container(border=True):
            if image_path:
                st.image(image_path, caption="📈 Generated Chart",
                         use_container_width=True)
            else:
                st.error("❌ Failed to generate the chart. Please try again.")


# ----------------- STREAMLIT TAB RENDER FUNCTION ----------------- #
def render_chart_tab():
    st.markdown("## 📊 Chart Generator")
    st.markdown(
        "Use this tool to **generate charts** from structured data or plain-text descriptions using AI.")

    input_text = st.text_area(
        "✏️ Input your data or description below", height=100)

    if st.button("📊 Generate Chart", key=1):
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
    """Display the chat interface with the given extracted text."""
    st.subheader("Ask Questions About Your Contract")

    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about your contract")

    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Display user message
        with st.chat_message("user"):
            st.write(user_question)

        # Get AI response
        with st.spinner("Thinking..."):
            ai_response = chat_with_document(extracted_text, user_question)

        # Add AI response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response
        })

        # Display AI response
        with st.chat_message("assistant"):
            st.write(ai_response)

        # Add a button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


def chat_tab():
    """Interface for chatting with contract document."""
    st.markdown("""
    This tab allows you to ask questions about the contract document you've uploaded.
    You can use a previously processed document or upload a new one. The system can analyze contracts in multiple languages.
    """)

    # Create placeholders for this tab
    modal_placeholder = st.empty()
    chat_placeholder = st.empty()

    # Check if we have previous extraction results
    has_previous_extraction = (
        'extracted_text' in st.session_state and
        st.session_state.extracted_text is not None
    )

    # Add radio button for document selection
    if has_previous_extraction:
        document_source = st.radio(
            "📄 Choose document source:",
            ["Use Previously Processed Document", "Upload New Document"],
            horizontal=True
        )
    else:
        document_source = "Upload New Document"

    if document_source == "Use Previously Processed Document":
        st.success("Using previously processed document for chat")
        show_chat_interface(st.session_state.extracted_text)
    else:
        # File upload for chat
        uploaded_file = st.file_uploader(
            "📁 Choose a contract document",
            type=["pdf", "png", "jpg", "jpeg"],
            key="chat_file_uploader"
        )

        if uploaded_file is not None:
            # Display file information
            file_type = uploaded_file.type
            file_name = os.path.splitext(uploaded_file.name)[
                0]  # Get filename without extension
            st.info(f"Uploaded file: {uploaded_file.name} ({file_type})")

            if st.button("🚀 Process for Chat", use_container_width=True):
                try:
                    # Show processing modal
                    logo_html = add_banner("assets/icon.png")
                    modal_html = create_modal(logo_html)
                    modal_placeholder.markdown(
                        modal_html, unsafe_allow_html=True)
                    extracted_text = []

                    # Process the document
                    with st.spinner("Processing document for chat..."):
                        file_bytes = uploaded_file.getvalue()

                        if file_type == "application/pdf":
                            status_text = st.empty()
                            status_text.text(
                                "Processing PDF with Azure Form Recognizer...")
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
                            # Use Azure Form Recognizer on the image
                            poller = document_client.begin_analyze_document(
                                model_id="prebuilt-read", document=file_bytes)
                            result = poller.result()

                            # Extract text from this page
                            page_text = []
                            for page in result.pages:
                                for line in page.lines:
                                    extracted_text.append({
                                        'text': line.content,
                                        'confidence': 0.98,
                                        'page': 1
                                    })

                        # Store in session state
                        st.session_state.extracted_text = extracted_text

                        # Clear chat history for a new document
                        if 'chat_history' in st.session_state:
                            st.session_state.chat_history = []

                        # Clear the modal
                        modal_placeholder.empty()

                        # Show success message and chat interface
                        st.success(
                            "Document processed! You can now chat with it.")
                        show_chat_interface(extracted_text)

                except Exception as e:
                    modal_placeholder.empty()
                    st.error(f"Error processing document: {str(e)}")


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
    """Display the chat interface for database with the given data."""
    st.subheader("Ask Questions About Your Contract Database")

    # Initialize chat history if not exists
    if 'db_chat_history' not in st.session_state:
        st.session_state.db_chat_history = []

    # Display chat history
    for message in st.session_state.db_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Ask a question about your contract database")

    if user_question:
        # Add user message to chat history
        st.session_state.db_chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Display user message
        with st.chat_message("user"):
            st.write(user_question)

        # Get AI response
        with st.spinner("Thinking..."):
            ai_response = chat_with_database(data, user_question)

        # Add AI response to chat history
        st.session_state.db_chat_history.append({
            "role": "assistant",
            "content": ai_response
        })

        # Display AI response
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
    """Interface for chatting with contract database from Azure SQL."""
    st.markdown("""
    This tab allows you to chat with a contract database stored in Azure SQL.
    """)
    # Show chat if data is already loaded
    if st.session_state.get("database_loaded", False):
        show_database_chat_interface(st.session_state["excel_data"])
        if st.button("📊 Generate Chart", key=2):
            text = st.session_state["db_chat_history"][-1]
            chart_ui = ChartUI()
            chart_ui.generate_and_render(text.get("content", ""))
        if st.button("🔄 Reset and Load Another Database"):
            st.session_state.pop("excel_data", None)
            st.session_state.pop("db_chat_history", None)
            st.session_state["database_loaded"] = False
            st.rerun()
        if st.button("🧹 Clear Chat History"):
            st.session_state.db_chat_history = []
            st.rerun()
        return

    # SQL query input (optional, or use default)
    query = st.text_area(
        "SQL Query to load contract data:",
        value="SELECT * FROM Contracts",
        help="Edit the SQL query if you want to filter or join tables."
    )
    if st.button("🔗 Load Database from Azure SQL", type="primary", use_container_width=True):
        with st.spinner("Loading data from Azure SQL..."):
            df = load_sql_database(query)
            if df is not None and not df.empty:
                st.session_state.excel_data = df
                st.session_state.database_loaded = True
                st.success(f"Loaded {len(df)} records from Azure SQL.")
                st.subheader("Database Preview")
                st.dataframe(df.head(5))
                show_database_chat_interface(df)
            else:
                st.error("No data loaded from Azure SQL.")


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
    # Create sidebar
    with st.sidebar:
        # Add logo
        try:
            st.image("logo.png", width=200)
        except:
            st.title("Contract Analysis Pro")

        # Add vertical space
        st.markdown("<br>" * 2, unsafe_allow_html=True)

        # Information section
        st.markdown("### About")
        st.markdown("""
        This application helps you extract and analyze information from contracts in multiple languages.
        Upload your documents and get instant insights!
        
        **Features**:
        - Extract key contract information in multiple languages (English, Spanish, Russian, Portuguese)
        - Process multiple files at once
        - Chat with your contract documents
        - Chat with contract database from SharePoint
        - Custom extraction templates
        """)

    # Create header
    st.title("Multilingual Contract Data Extraction")
    st.markdown(
        "Extract, analyze, and chat with your contract documents in multiple languages")

    # Create tabs with icons
    tab1, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Standard Extraction",
        "💬 Chat with Contract",
        "🗃️ Chat with Contract Database",
        "📊 Generate Charts",
        "📈 Contracts Report"
    ])

    # Tab 1: Standard extraction
    with tab1:
        st.markdown("""
            ### Standard Contract Information Extraction
            Upload your contracts to extract key information automatically in multiple languages.
            
            You can upload multiple files and process them all at once.
        """)

        # Add language selector
        selected_language = st.selectbox(
            "🌐 Select language for extraction:",
            options=list(SYSTEM_PROMPTS.keys()),
            index=0  # Default to English
        )

        # Brief explanation of language selection
        st.info(
            f"Selected language: {selected_language}. The extraction system will use prompts and respond in this language.")

        # File uploader with modern styling
        uploaded_files = st.file_uploader(
            "📁 Drag and drop your files here",
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
                        <div class="uploadedFile">
                            <p><strong>{file.name}</strong></p>
                            <p>Size: {file.size / 1024:.1f} KB</p>
                            <p>Type: {file.type}</p>
                        </div>
                    """, unsafe_allow_html=True)

            if st.button("🚀 Process All Documents", use_container_width=True):
                # Initialize list to store all results
                all_results = []

                # Process files and show results
                with st.spinner("Processing all documents..."):
                    for file_index, file in enumerate(uploaded_files):
                        try:
                            st.markdown(
                                f"### Processing file {file_index + 1}/{len(uploaded_files)}: {file.name}")
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
                                with st.expander(f"Results for {file.name}", expanded=False):
                                    # Display results
                                    # Instead of calling display_extraction_results which uses expanders,
                                    # we'll show the results directly to avoid nested expanders
                                    st.subheader(
                                        "Extracted Contract Information")

                                    # Show page info if multi-page document
                                    if metrics.get("page_count", 1) > 1:
                                        st.info(
                                            f"Processed {metrics['page_count']} pages")

                                        # Display thumbnails of processed images if available
                                        if metrics.get("saved_images"):
                                            st.subheader("Processed Pages")
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
                                            f"### 📄 Contract Payment Type {idx + 1}")

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
                                            value = contract.get(
                                                key, "Not found")
                                            col.markdown(
                                                f"**{label}:** {value}")

                                        # List-based fields
                                        list_fields = [
                                            ("Promo Tactic", "Promo Tactic(s)"),
                                            ("Product Category",
                                             "Product Category(ies)"),
                                            ("Incentives Details",
                                             "Incentives Details"),
                                            ("Legal Aspects", "Legal Aspects")
                                        ]
                                        for key, label in list_fields:
                                            items = contract.get(key, [])
                                            if isinstance(items, list) and items:
                                                st.markdown(f"**{label}:**")
                                                for item in items:
                                                    st.markdown(f"- {item}")
                                            else:
                                                st.markdown(
                                                    f"**{label}:** Not found")

                                        # Multi-line text fields
                                        long_fields = [
                                            ("Promotion Display",
                                             "Promotion Display"),
                                            ("Payment Structure",
                                             "Payment Structure"),
                                            ("Penalties", "Penalties")
                                        ]
                                        for key, label in long_fields:
                                            st.markdown(
                                                f"**{label}:** {contract.get(key, 'Not found')}")

                                    # Show tabs for additional information
                                    tab1, tab2, tab3 = st.tabs(
                                        ["Raw JSON", "OCR Text", "Processing Metrics"])

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
                                        # Remove image paths from display
                                        display_metrics = {
                                            k: v for k, v in metrics.items() if k != "saved_images"}
                                        st.json(display_metrics)

                                    # Add download buttons for this specific file
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        json_str = json.dumps(
                                            openai_results, indent=2)
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
                                                    "File": file.name,
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

                            else:
                                st.error(
                                    f"Processing failed for {file.name}. Skipping to next file.")
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")

                # If we have processed multiple files, show a summary and bulk download options
                if len(all_results) > 1:
                    st.markdown("### Batch Processing Summary")
                    st.success(
                        f"Successfully processed {len(all_results)} of {len(uploaded_files)} document(s)!")

                    # Add bulk download options
                    st.markdown("### Download All Results")
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
                            label="📥 Download All Results (JSON)",
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
                            label="📥 Download All Results (CSV)",
                            data=csv_string,
                            file_name=csv_filename,
                            mime="text/csv",
                            use_container_width=True
                        )

                elif len(all_results) == 1:
                    st.success(f"Successfully processed the document!")

                # Add comparative analysis option if multiple documents
                if len(all_results) > 1:
                    st.markdown("### Comparative Analysis")
                    with st.expander("View Field Comparison Across Documents"):
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
            ### Chat with Your Contract
            Ask questions about your contract and get instant answers.
        """)

        if 'extracted_text' in st.session_state:
            document_source = st.radio(
                "📄 Choose document source:",
                ["Use Previously Processed Document", "Upload New Document"],
                horizontal=True
            )

            if document_source == "Use Previously Processed Document":
                show_chat_interface(st.session_state.extracted_text)
            else:
                uploaded_file = st.file_uploader(
                    "📁 Choose a contract document",
                    type=["pdf", "png", "jpg", "jpeg"],
                    key="chat_file_uploader"
                )

                if uploaded_file and st.button("🚀 Process for Chat", use_container_width=True):
                    try:
                        with st.spinner("Processing document for chat..."):
                            file_bytes = uploaded_file.getvalue()
                            file_type = uploaded_file.type
                            file_name = os.path.splitext(uploaded_file.name)[0]

                            extracted_text = []

                            if file_type == "application/pdf":
                                status_text = st.empty()
                                status_text.text(
                                    "Processing PDF with Azure Form Recognizer...")
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
                                # Use Azure Form Recognizer on the image
                                poller = document_client.begin_analyze_document(
                                    model_id="prebuilt-read", document=file_bytes)
                                result = poller.result()

                                # Extract text from this page
                                page_text = []
                                for page in result.pages:
                                    for line in page.lines:
                                        extracted_text.append({
                                            'text': line.content,
                                            'confidence': 0.98,
                                            'page': 1
                                        })

                            # Store in session state
                            st.session_state.extracted_text = extracted_text

                            # Clear chat history for a new document
                            if 'chat_history' in st.session_state:
                                st.session_state.chat_history = []

                            # Show success message and chat interface
                            st.success(
                                "Document processed! You can now chat with it.")
                            show_chat_interface(extracted_text)

                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        else:
            uploaded_file = st.file_uploader(
                "📁 Choose a contract document",
                type=["pdf", "png", "jpg", "jpeg"],
                key="chat_file_uploader"
            )

            if uploaded_file and st.button("🚀 Process for Chat", use_container_width=True):
                try:
                    with st.spinner("Processing document for chat..."):
                        file_bytes = uploaded_file.getvalue()
                        file_type = uploaded_file.type
                        file_name = os.path.splitext(uploaded_file.name)[0]
                        extracted_text = []

                        # Process the document and extract text
                        if file_type == "application/pdf":
                            status_text = st.empty()
                            status_text.text(
                                "Processing PDF with Azure Form Recognizer...")
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
                            # Use Azure Form Recognizer on the image
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

                        # Store in session state
                        st.session_state.extracted_text = extracted_text

                        # Clear chat history for a new document
                        if 'chat_history' in st.session_state:
                            st.session_state.chat_history = []

                        # Show success message and chat interface
                        st.success(
                            "Document processed! You can now chat with it.")
                        show_chat_interface(extracted_text)

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

    # Tab 4: Chat with database
    with tab4:
        st.markdown("""
            ### Chat with Contract Database
            Connect to the Azure SQL database and chat with your contract data.
        """)
        chat_database_tab()

    # Tab 5: Generate charts
    with tab5:
        render_chart_tab()

    with tab6:
        st.markdown("""
            ### Power BI Report
            Power BI Report. You must be log in to Power BI to see the report.
            If you are not logged in, you will be redirected to the login page.
        """)
        powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiZGZjMGEzOTUtOTUxZi00NDRiLWIyZTgtNDkzZjM4MDVkOGI0IiwidCI6IjBlNTczMTNlLTc3ZWUtNDlhYS05ZDc2LTdmOWZjOTEyMWE3YSJ9"
        st.markdown(
            f"""
            <iframe
                title="Power BI Report"
                width="100%"
                height="800"
                src="{powerbi_url}"
                frameborder="0"
                allowFullScreen="true">
            </iframe>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
