# Auto-generated from app.py prompts
SYSTEM_PROMPT_ENGLISH = '''
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
 
'''

SYSTEM_PROMPT_SPANISH = '''
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
'''

SYSTEM_PROMPT_RUSSIAN = '''
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
'''

SYSTEM_PROMPT_PORTUGUESE = '''
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
'''

SYSTEM_PROMPTS = {
    "English": SYSTEM_PROMPT_ENGLISH,
    "Spanish": SYSTEM_PROMPT_SPANISH,
    "Russian": SYSTEM_PROMPT_RUSSIAN,
    "Portuguese": SYSTEM_PROMPT_PORTUGUESE,
}
DEFAULT_PROMPT = SYSTEM_PROMPT_ENGLISH
