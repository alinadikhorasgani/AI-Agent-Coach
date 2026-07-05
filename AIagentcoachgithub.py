"""
AI Nutrition Coach with LangGraph + Gradio

Main requirements covered:
1. The PDF is generated from the exact same final_report string shown in chat.
2. The final report always ends with a PubMed Article Analysis section.
3. The chatbot includes explicit Guardrails to avoid unsafe nutrition/medical advice.
4. API keys are loaded from environment variables, not hard-coded.

Install:
    pip install gradio langchain-openai langgraph requests reportlab

Environment variables:
    export FDC_API_KEY="your_usda_fdc_api_key"
    export LLM_API_KEY="your_llm_api_key"
    export LLM_BASE_URL="https://api.gapapi.com/v1"
    export LLM_MODEL="deepseek-chat"

Optional for NCBI:
    export NCBI_API_KEY="your_ncbi_api_key"
    export NCBI_EMAIL="your_email@example.com"
"""

import os
import re
import json
import time
import html
import hashlib
import requests
import gradio as gr

from typing import Any, Dict, List, Optional, TypedDict
from xml.etree import ElementTree as ET

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================================================
# 1) CONFIGURATION
# =========================================================

FDC_API_KEY = os.getenv("FDC_API_KEY", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.gapapi.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_TOOL_NAME = "ai_nutrition_coach"

if not LLM_API_KEY:
    raise ValueError("Missing LLM_API_KEY. Please set it as an environment variable.")

llm = ChatOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    temperature=0.2,
)


# =========================================================
# 2) GUARDRAILS
# =========================================================

GUARDRAILS_SYSTEM_PROMPT = """
You are an educational AI nutrition coach.

You must follow these guardrails:
- Do not diagnose, treat, or claim to cure any disease.
- Do not replace a physician, registered dietitian, therapist, or emergency care.
- Do not recommend extreme calorie restriction, starvation, purging, laxative misuse, diet pills, or unsafe rapid weight loss.
- Do not provide meal plans that are dangerously low in calories or nutritionally incomplete.
- Do not encourage body shaming, appearance-based pressure, or comparison with others.
- If the user is under 18, focus on healthy habits, growth, sleep, movement, hydration, and family/clinician support. Do not prescribe weight-loss calorie targets.
- If the user reports pregnancy, diabetes, kidney disease, eating disorder history, severe allergies, or other significant medical conditions, advise consultation with a qualified clinician.
- For urgent symptoms, advise seeking urgent medical help.
- Use PubMed research cautiously. Explain that article findings support education and discussion with clinicians, not personal medical treatment.
- Keep the tone supportive, evidence-aware, and safe.
"""

UNSAFE_INPUT_PATTERNS = [
    r"\bstarv(e|ing|ation)\b",
    r"\bpurge\b",
    r"\bvomit\b",
    r"\blaxative\b",
    r"\bdiet pill\b",
    r"\bwater fast\b",
    r"\bnot eating\b",
    r"\bskip all meals\b",
    r"\blose\s+\d+\s*(kg|kilograms|pounds|lbs)\s+in\s+\d+\s*(day|days|week|weeks)\b",
    r"\b500\s*calories\b",
    r"\b800\s*calories\b",
]

GUARDRAIL_RESPONSE = (
    "I cannot help with unsafe dieting, extreme restriction, purging, laxative misuse, "
    "or rapid weight-loss tactics. I can help with safer nutrition education, balanced meals, "
    "general lifestyle habits, and questions to discuss with a qualified clinician."
)


class GuardrailService:
    @staticmethod
    def check_user_input(text: str) -> Optional[str]:
        normalized = text.lower().strip()
        for pattern in UNSAFE_INPUT_PATTERNS:
            if re.search(pattern, normalized):
                return GUARDRAIL_RESPONSE
        return None

    @staticmethod
    def safe_llm_call(user_prompt: str) -> str:
        response = llm.invoke(
            [
                SystemMessage(content=GUARDRAILS_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        return response.content.strip()


# =========================================================
# 3) STATE
# =========================================================

class CoachState(TypedDict):
    user_profile: Dict[str, str]
    step: int
    meal_analysis: Optional[Dict[str, Any]]
    body_metrics: Optional[Dict[str, Any]]
    research_articles: List[Dict[str, Any]]
    final_report: str
    report_hash: Optional[str]
    pdf_path: Optional[str]
    messages: List[BaseMessage]
    is_finished: bool


def create_initial_state() -> CoachState:
    return {
        "user_profile": {},
        "step": 0,
        "meal_analysis": None,
        "body_metrics": None,
        "research_articles": [],
        "final_report": "",
        "report_hash": None,
        "pdf_path": None,
        "messages": [],
        "is_finished": False,
    }


# =========================================================
# 4) QUESTIONS
# =========================================================

QUESTIONS = [
    ("age", "1. What is your age?"),
    ("gender", "2. What is your sex? Male, female, or prefer not to say."),
    ("height", "3. What is your height in centimeters?"),
    ("weight", "4. What is your weight in kilograms?"),
    ("activity", "5. What is your activity level? Sedentary, light, moderate, active, or very active."),
    ("goal", "6. What is your main nutrition goal?"),
    ("conditions", "7. Do you have any medical conditions or important health notes? If none, write 'none'."),
    ("restrictions", "8. Do you have any dietary restrictions, allergies, or preferences?"),
    ("meal", "9. Describe one typical meal with approximate amounts, or write 'none'."),
]


# =========================================================
# 5) UTILITY FUNCTIONS
# =========================================================

def safe_float(text: str) -> float:
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(text))
    return float(match.group()) if match else 0.0


def normalize_activity_multiplier(activity: str) -> float:
    value = activity.lower()
    if "very" in value:
        return 1.725
    if "active" in value:
        return 1.55
    if "moderate" in value:
        return 1.55
    if "light" in value:
        return 1.375
    return 1.2


def interpret_bmi(age: float, bmi: float) -> str:
    if bmi <= 0:
        return "BMI could not be calculated from the provided data."
    if age < 18:
        return (
            "For users under 18, adult BMI categories are not appropriate. "
            "BMI-for-age percentile should be interpreted by a qualified clinician."
        )
    if bmi < 18.5:
        return "Adult BMI category: underweight range."
    if bmi < 25:
        return "Adult BMI category: usual range."
    if bmi < 30:
        return "Adult BMI category: overweight range."
    return "Adult BMI category: obesity range."


def format_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\[.*\]", text, flags=re.S)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def flatten_xml_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def clean_pubmed_query_part(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:100].strip()


# =========================================================
# 6) USDA FOODDATA CENTRAL SERVICE
# =========================================================

class USDANutrientService:
    BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

    @staticmethod
    def parse_meal_text(text: str) -> List[Dict[str, Any]]:
        if not text or text.lower().strip() in ["none", "no", "skip"]:
            return []

        items = []
        chunks = text.replace("،", ",").split(",")

        for chunk in chunks:
            chunk = chunk.strip()
            grams_match = re.search(r"(\d+(\.\d+)?)\s*(g|gram|grams)", chunk, re.I)
            grams = float(grams_match.group(1)) if grams_match else 100.0
            name = re.sub(r"(\d+(\.\d+)?)\s*(g|gram|grams)", "", chunk, flags=re.I).strip()

            if name:
                items.append({"query": name, "grams": grams})

        return items

    @staticmethod
    def nutrient_value(food: Dict[str, Any], nutrient_id: int, grams: float) -> float:
        for nutrient in food.get("foodNutrients", []):
            if nutrient.get("nutrientId") == nutrient_id:
                value_per_100g = float(nutrient.get("value", 0))
                return value_per_100g * grams / 100
        return 0.0

    @classmethod
    def get_nutrition(cls, text: str) -> Optional[Dict[str, Any]]:
        items = cls.parse_meal_text(text)
        if not items:
            return None

        if not FDC_API_KEY:
            return {
                "warning": "FDC_API_KEY is missing. Meal nutrition could not be calculated.",
                "totals": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0},
                "items": [],
            }

        totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        details = []

        for item in items:
            try:
                response = requests.post(
                    cls.BASE_URL,
                    params={"api_key": FDC_API_KEY},
                    json={"query": item["query"], "pageSize": 1},
                    timeout=10,
                )
                response.raise_for_status()
                foods = response.json().get("foods", [])

                if not foods:
                    details.append(
                        {
                            "query": item["query"],
                            "grams": item["grams"],
                            "matched_food": None,
                            "note": "No USDA match found.",
                        }
                    )
                    continue

                food = foods[0]
                grams = item["grams"]

                calories = cls.nutrient_value(food, 1008, grams)
                protein = cls.nutrient_value(food, 1003, grams)
                fat = cls.nutrient_value(food, 1004, grams)
                carbs = cls.nutrient_value(food, 1005, grams)

                totals["calories"] += calories
                totals["protein"] += protein
                totals["fat"] += fat
                totals["carbs"] += carbs

                details.append(
                    {
                        "query": item["query"],
                        "matched_food": food.get("description", "Unknown food"),
                        "grams": grams,
                        "calories": round(calories, 1),
                        "protein": round(protein, 1),
                        "carbs": round(carbs, 1),
                        "fat": round(fat, 1),
                    }
                )

            except Exception as error:
                details.append(
                    {
                        "query": item["query"],
                        "grams": item["grams"],
                        "matched_food": None,
                        "note": f"USDA lookup failed: {str(error)}",
                    }
                )

        return {
            "totals": {key: round(value, 1) for key, value in totals.items()},
            "items": details,
        }


# =========================================================
# 7) PUBMED SERVICE
# =========================================================

class PubMedService:
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    @staticmethod
    def common_params() -> Dict[str, str]:
        params = {"tool": NCBI_TOOL_NAME}
        if NCBI_EMAIL:
            params["email"] = NCBI_EMAIL
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        return params

    @classmethod
    def search_pubmed_ids(cls, term: str, retmax: int = 10) -> List[str]:
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": str(retmax),
            "sort": "relevance",
            "retmode": "json",
            **cls.common_params(),
        }

        response = requests.get(cls.ESEARCH_URL, params=params, timeout=10)
        response.raise_for_status()

        return response.json().get("esearchresult", {}).get("idlist", [])

    @classmethod
    def fetch_article_details(cls, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            **cls.common_params(),
        }

        response = requests.get(cls.EFETCH_URL, params=params, timeout=15)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        articles = []

        for pubmed_article in root.findall(".//PubmedArticle"):
            medline = pubmed_article.find("./MedlineCitation")
            article_node = medline.find("./Article") if medline is not None else None

            pmid = flatten_xml_text(medline.find("./PMID")) if medline is not None else ""

            title = flatten_xml_text(article_node.find("./ArticleTitle")) if article_node is not None else "Untitled article"

            abstract_parts = []
            if article_node is not None:
                for abstract_text in article_node.findall("./Abstract/AbstractText"):
                    label = abstract_text.attrib.get("Label", "")
                    text = flatten_xml_text(abstract_text)
                    if text:
                        abstract_parts.append(f"{label}: {text}" if label else text)

            abstract = " ".join(abstract_parts)

            journal = ""
            year = ""

            if article_node is not None:
                journal = flatten_xml_text(article_node.find("./Journal/Title"))
                year_node = article_node.find("./Journal/JournalIssue/PubDate/Year")
                medline_date_node = article_node.find("./Journal/JournalIssue/PubDate/MedlineDate")

                year = flatten_xml_text(year_node)
                if not year:
                    medline_date = flatten_xml_text(medline_date_node)
                    year_match = re.search(r"\d{4}", medline_date)
                    year = year_match.group(0) if year_match else ""

            authors = []
            if article_node is not None:
                for author in article_node.findall("./AuthorList/Author")[:4]:
                    last = flatten_xml_text(author.find("./LastName"))
                    initials = flatten_xml_text(author.find("./Initials"))
                    full_name = f"{last} {initials}".strip()
                    if full_name:
                        authors.append(full_name)

            doi = ""
            article_id_list = pubmed_article.find("./PubmedData/ArticleIdList")
            if article_id_list is not None:
                for article_id in article_id_list.findall("./ArticleId"):
                    if article_id.attrib.get("IdType") == "doi":
                        doi = flatten_xml_text(article_id)
                        break

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "authors": authors,
                    "doi": doi,
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                }
            )

        articles.sort(key=lambda item: bool(item.get("abstract")), reverse=True)
        return articles

    @classmethod
    def fetch_research(cls, goal: str, condition: str, max_articles: int = 3) -> List[Dict[str, Any]]:
        safe_goal = clean_pubmed_query_part(goal) or "healthy diet"
        safe_condition = clean_pubmed_query_part(condition)

        base_terms = [f"({safe_goal})", "(nutrition OR diet OR lifestyle)"]

        if safe_condition and safe_condition.lower() not in ["none", "no", "nothing", "n/a"]:
            base_terms.append(f"({safe_condition})")

        primary_query = " AND ".join(base_terms)

        try:
            pmids = cls.search_pubmed_ids(primary_query, retmax=10)

            if len(pmids) < max_articles:
                fallback_query = f"({safe_goal}) AND (nutrition OR diet OR lifestyle)"
                fallback_pmids = cls.search_pubmed_ids(fallback_query, retmax=10)
                pmids = list(dict.fromkeys(pmids + fallback_pmids))

            articles = cls.fetch_article_details(pmids)
            return articles[:max_articles]

        except Exception as error:
            print(f"PubMed error: {error}")
            return []


# =========================================================
# 8) PDF SERVICE
# =========================================================

class PDFReportService:
    @staticmethod
    def register_font() -> str:
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/local/share/fonts/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ]

        for font_path in font_candidates:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
                return "DejaVuSans"

        return "Helvetica"

    @classmethod
    def build_pdf_from_exact_report(cls, report_text: str, output_path: str) -> None:
        """
        Important:
        The PDF is generated directly from final_report.
        Do not summarize, truncate, rewrite, or call the LLM again here.
        """

        font_name = cls.register_font()

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title="AI Nutrition Coach Report",
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=16,
            leading=20,
            spaceAfter=10,
        )

        heading_style = ParagraphStyle(
            "ReportHeading",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=12,
            leading=16,
            spaceBefore=8,
            spaceAfter=5,
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=13,
            spaceAfter=5,
        )

        story = []

        lines = report_text.splitlines()

        for index, line in enumerate(lines):
            clean_line = html.escape(line)

            if not line.strip():
                story.append(Spacer(1, 5))
                continue

            if index == 0:
                story.append(Paragraph(clean_line, title_style))
            elif line.isupper() or re.match(r"^\d+\.\s+[A-Z]", line.strip()):
                story.append(Paragraph(clean_line, heading_style))
            else:
                story.append(Paragraph(clean_line, body_style))

        doc.build(story)


# =========================================================
# 9) REPORT FORMATTING HELPERS
# =========================================================

def format_meal_analysis_for_prompt(meal_analysis: Optional[Dict[str, Any]]) -> str:
    if not meal_analysis:
        return "No meal analysis was provided."

    return format_json(meal_analysis)


def build_pubmed_section(articles: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("PUBMED ARTICLE ANALYSIS")
    lines.append(
        "This section summarizes PubMed articles for educational context only. "
        "It should not be used as diagnosis, treatment, or a substitute for clinician guidance."
    )

    if not articles:
        lines.append("")
        lines.append(
            "No PubMed articles could be retrieved. The user should consult a qualified professional "
            "for evidence-based guidance."
        )
        return "\n".join(lines)

    for index, article in enumerate(articles[:3], start=1):
        lines.append("")
        lines.append(f"Article {index}")
        lines.append(f"Title: {article.get('title', 'Untitled article')}")
        lines.append(f"PMID: {article.get('pmid', 'Unavailable')}")
        lines.append(f"Journal: {article.get('journal', 'Unavailable')}")
        lines.append(f"Year: {article.get('year', 'Unavailable')}")
        lines.append(f"Authors: {', '.join(article.get('authors', [])) or 'Unavailable'}")
        lines.append(f"PubMed URL: {article.get('pubmed_url', 'Unavailable')}")

        lines.append(f"Context: {article.get('objective_or_context', 'Not available.')}")
        lines.append(f"Key finding: {article.get('key_finding', 'Not available.')}")
        lines.append(f"Practical implication: {article.get('practical_implication', 'Not available.')}")
        lines.append(f"Limitation or caution: {article.get('limitation', 'Not available.')}")

    return "\n".join(lines)


def build_report_hash(report_text: str) -> str:
    return hashlib.sha256(report_text.encode("utf-8")).hexdigest()


# =========================================================
# 10) GRAPH NODES
# =========================================================

def info_collector_node(state: CoachState) -> Dict[str, Any]:
    messages = state["messages"]
    step = state["step"]

    if state.get("pdf_path"):
        return {
            "messages": messages
            + [
                AIMessage(
                    content=(
                        "The report has already been generated. "
                        "Please use the Reset button if you want to start a new report."
                    )
                )
            ]
        }

    if not messages:
        return {
            "messages": [
                AIMessage(content="Welcome to the AI Nutrition Coach. Let's begin.\n\n" + QUESTIONS[0][1])
            ],
            "step": 0,
        }

    last_message = messages[-1]

    if isinstance(last_message, HumanMessage):
        guardrail_message = GuardrailService.check_user_input(last_message.content)
        if guardrail_message:
            return {
                "messages": messages + [AIMessage(content=guardrail_message)],
                "step": step,
            }

        new_profile = dict(state["user_profile"])
        question_key = QUESTIONS[step][0]
        new_profile[question_key] = last_message.content.strip()

        if step + 1 < len(QUESTIONS):
            next_question = QUESTIONS[step + 1][1]
            return {
                "user_profile": new_profile,
                "step": step + 1,
                "messages": messages + [AIMessage(content=next_question)],
            }

        return {
            "user_profile": new_profile,
            "is_finished": True,
            "messages": messages
            + [
                AIMessage(
                    content=(
                        "Thank you. I have collected the required information and will generate "
                        "the nutrition report, including a PubMed article analysis section."
                    )
                )
            ],
        }

    return {}


def analyzer_node(state: CoachState) -> Dict[str, Any]:
    profile = state["user_profile"]

    age = safe_float(profile.get("age", "0"))
    weight = safe_float(profile.get("weight", "0"))
    height = safe_float(profile.get("height", "0"))

    bmi = round(weight / ((height / 100) ** 2), 1) if height > 0 and weight > 0 else 0

    gender = profile.get("gender", "").lower()
    sex_adjustment = 5 if "male" in gender else -161

    bmr = 0
    tdee = 0

    if age > 0 and weight > 0 and height > 0:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + sex_adjustment
        tdee = bmr * normalize_activity_multiplier(profile.get("activity", ""))

    body_metrics = {
        "age": age,
        "bmi": bmi,
        "bmi_note": interpret_bmi(age, bmi),
        "bmr": round(bmr),
        "estimated_tdee": round(tdee),
        "tdee_note": (
            "Estimated energy needs are approximate and should not be treated as a prescription."
        ),
    }

    meal_analysis = USDANutrientService.get_nutrition(profile.get("meal", ""))

    return {
        "body_metrics": body_metrics,
        "meal_analysis": meal_analysis,
    }


def researcher_node(state: CoachState) -> Dict[str, Any]:
    profile = state["user_profile"]

    raw_articles = PubMedService.fetch_research(
        goal=profile.get("goal", "healthy diet"),
        condition=profile.get("conditions", "none"),
        max_articles=3,
    )

    if not raw_articles:
        return {"research_articles": []}

    article_payload = [
        {
            "pmid": article.get("pmid"),
            "title": article.get("title"),
            "abstract": article.get("abstract"),
            "journal": article.get("journal"),
            "year": article.get("year"),
            "authors": article.get("authors"),
            "doi": article.get("doi"),
            "pubmed_url": article.get("pubmed_url"),
        }
        for article in raw_articles
    ]

    prompt = f"""
Analyze the following PubMed articles for an educational nutrition coaching report.

Return only a valid JSON array.
Return exactly one JSON object per article.
Do not invent findings that are not supported by the title or abstract.
If an abstract is missing, say that the evidence cannot be assessed from metadata alone.

Required fields for each object:
- pmid
- title
- journal
- year
- authors
- pubmed_url
- objective_or_context
- key_finding
- practical_implication
- limitation

Articles:
{format_json(article_payload)}
"""

    try:
        response_text = GuardrailService.safe_llm_call(prompt)
        analyzed_articles = extract_json_array(response_text)

        if not analyzed_articles:
            raise ValueError("The LLM did not return valid JSON.")

        merged = []
        for raw, analyzed in zip(raw_articles, analyzed_articles):
            merged.append({**raw, **analyzed})

        return {"research_articles": merged[:3]}

    except Exception as error:
        print(f"Research analysis error: {error}")

        fallback_articles = []
        for article in raw_articles[:3]:
            fallback_articles.append(
                {
                    **article,
                    "objective_or_context": (
                        "This article was retrieved from PubMed based on the user's goal and health context."
                    ),
                    "key_finding": (
                        "A reliable finding could not be summarized automatically. "
                        "The article should be reviewed directly."
                    ),
                    "practical_implication": (
                        "Use this article as a starting point for discussion with a qualified clinician."
                    ),
                    "limitation": (
                        "Automated analysis was unavailable or incomplete, so no personalized conclusion should be drawn."
                    ),
                }
            )

        return {"research_articles": fallback_articles}


def reporter_node(state: CoachState) -> Dict[str, Any]:
    profile = state["user_profile"]
    body_metrics = state["body_metrics"] or {}
    meal_analysis = state["meal_analysis"]
    pubmed_section = build_pubmed_section(state["research_articles"])

    age = safe_float(profile.get("age", "0"))
    minor_instruction = ""
    if age and age < 18:
        minor_instruction = (
            "The user is under 18. Do not provide weight-loss calorie targets. "
            "Focus on balanced habits, growth, energy, sleep, hydration, and clinician/family support."
        )

    prompt = f"""
Create a complete English plain-text nutrition coaching report.

Important formatting rules:
- Do not use markdown tables.
- Do not include a PubMed section. It will be appended after your report body by the program.
- Use clear section headings.
- Keep the report educational, safe, practical, and non-diagnostic.
- Include a medical disclaimer.
- Do not prescribe treatment.
- Do not recommend extreme dieting.
- Do not shame the user.
- {minor_instruction}

User profile:
{format_json(profile)}

Calculated body metrics:
{format_json(body_metrics)}

Meal analysis:
{format_meal_analysis_for_prompt(meal_analysis)}

Write the report body now.
"""

    try:
        report_body = GuardrailService.safe_llm_call(prompt).strip()
    except Exception as error:
        print(f"Report generation error: {error}")
        report_body = (
            "AI NUTRITION COACH REPORT\n\n"
            "The report could not be generated because the language model request failed.\n\n"
            "This tool provides educational nutrition information only and does not replace medical care."
        )

    final_report = report_body + "\n\n" + pubmed_section
    report_hash = build_report_hash(final_report)

    assistant_message = (
        final_report
        + "\n\n"
        + f"Report content SHA-256: {report_hash}\n"
        + "The PDF is generated from this exact final_report string."
    )

    return {
        "final_report": final_report,
        "report_hash": report_hash,
        "messages": state["messages"] + [AIMessage(content=assistant_message)],
    }


def pdf_generator_node(state: CoachState) -> Dict[str, Any]:
    final_report = state.get("final_report", "")

    if not final_report:
        return {"pdf_path": None}

    os.makedirs("reports", exist_ok=True)

    timestamp = int(time.time())
    pdf_path = os.path.abspath(f"reports/ai_nutrition_report_{timestamp}.pdf")
    txt_path = os.path.abspath(f"reports/ai_nutrition_report_{timestamp}.txt")

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(final_report)

    try:
        PDFReportService.build_pdf_from_exact_report(final_report, pdf_path)
        return {"pdf_path": pdf_path}

    except Exception as error:
        print(f"PDF generation error: {error}")
        return {"pdf_path": None}


# =========================================================
# 11) GRAPH CONSTRUCTION
# =========================================================

def route_after_gather_info(state: CoachState) -> str:
    if state.get("is_finished") and not state.get("final_report"):
        return "analyze"
    return "end"


builder = StateGraph(CoachState)

builder.add_node("gather_info", info_collector_node)
builder.add_node("analyze", analyzer_node)
builder.add_node("research", researcher_node)
builder.add_node("report", reporter_node)
builder.add_node("pdf", pdf_generator_node)

builder.set_entry_point("gather_info")

builder.add_conditional_edges(
    "gather_info",
    route_after_gather_info,
    {
        "analyze": "analyze",
        "end": END,
    },
)

builder.add_edge("analyze", "research")
builder.add_edge("research", "report")
builder.add_edge("report", "pdf")
builder.add_edge("pdf", END)

coach_app = builder.compile()


# =========================================================
# 12) GRADIO UI
# =========================================================

def messages_to_gradio(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    formatted = []

    for message in messages:
        if isinstance(message, HumanMessage):
            formatted.append({"role": "user", "content": message.content})
        else:
            formatted.append({"role": "assistant", "content": message.content})

    return formatted


def start_chat(state: Optional[CoachState]):
    new_state = create_initial_state()
    new_state = coach_app.invoke(new_state)

    return messages_to_gradio(new_state["messages"]), new_state, None


def reset_chat():
    new_state = create_initial_state()
    new_state = coach_app.invoke(new_state)

    return "", messages_to_gradio(new_state["messages"]), new_state, None


def chat_fn(user_input: str, history: List[Dict[str, str]], state: Optional[CoachState]):
    if state is None:
        state = create_initial_state()
        state = coach_app.invoke(state)

    user_input = user_input.strip() if user_input else ""

    if not user_input:
        return "", messages_to_gradio(state["messages"]), state, state.get("pdf_path")

    state["messages"].append(HumanMessage(content=user_input))

    try:
        new_state = coach_app.invoke(state)
    except Exception as error:
        print(f"Graph error: {error}")
        state["messages"].append(
            AIMessage(
                content=(
                    "An internal error occurred while processing the report. "
                    "Please check the server logs and try again."
                )
            )
        )
        new_state = state

    return "", messages_to_gradio(new_state["messages"]), new_state, new_state.get("pdf_path")


with gr.Blocks() as demo:
    gr.Markdown("# AI Nutrition Coach")
    gr.Markdown(
        "This tool provides educational nutrition guidance only. "
        "It does not replace medical care, diagnosis, or treatment."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, type="messages")
            msg = gr.Textbox(
                placeholder="Type your answer here...",
                label="Message",
            )

            with gr.Row():
                reset_button = gr.Button("Reset")

        with gr.Column(scale=1):
            file_output = gr.File(label="Download PDF Report")

    state_store = gr.State()

    demo.load(
        start_chat,
        inputs=[state_store],
        outputs=[chatbot, state_store, file_output],
    )

    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, state_store],
        outputs=[msg, chatbot, state_store, file_output],
    )

    reset_button.click(
        reset_chat,
        inputs=[],
        outputs=[msg, chatbot, state_store, file_output],
    )


if __name__ == "__main__":
    demo.launch(share=True)
