# 🥗 AI Nutrition Coach (LangGraph + PubMed + PDF Report System)

## Overview

This project is an **AI-powered Nutrition Coaching system** built using:

* LangGraph (workflow orchestration)
* OpenAI-compatible LLM API
* USDA FoodData Central API
* PubMed (NCBI E-utilities)
* Gradio (UI)
* ReportLab (PDF generation)

It collects user health data, analyzes nutrition, retrieves scientific evidence from PubMed, and generates a **structured AI report + downloadable PDF**.

---

## Key Features

### 1. Interactive Nutrition Interview

The system asks step-by-step questions:

* Age, gender, height, weight
* Activity level
* Health goals
* Medical conditions
* Dietary restrictions
* Typical meals

---

### 2. Body Metrics Calculation

Automatically computes:

* BMI (Body Mass Index)
* BMR (Basal Metabolic Rate)
* TDEE (Total Daily Energy Expenditure)

---

### 3. Meal Nutrient Analysis (USDA API)

Uses **FoodData Central API** to estimate:

* Calories
* Protein
* Carbohydrates
* Fat

---

### 4. PubMed Research Integration

Fetches **3 relevant scientific papers** using NCBI E-utilities and:

* Extracts title, abstract, authors
* Analyzes research relevance using LLM
* Generates structured insights:

  * Key findings
  * Practical implications
  * Limitations

---

### 5. AI Report Generation

The LLM generates a structured nutrition report including:

* Body metrics summary
* Nutrition analysis
* Personalized educational insights
* Safety disclaimer
* PubMed research section (mandatory at the end)

---

### 6. PDF Export (Exact Mirror Output)

* PDF is generated **directly from the final LLM report**
* No re-generation or summarization
* Ensures consistency between chat output and downloadable file

---

### 7. Guardrails System (Safety Layer)

The system includes strict guardrails to prevent unsafe outputs:

#### Blocks:

* Extreme dieting advice
* Starvation or fasting abuse
* Purging or laxative misuse
* Unsafe rapid weight loss
* Medical diagnosis claims
* Harmful body image content

#### Behavior:

* Detects unsafe input patterns
* Replaces response with safe educational message
* Enforces LLM system-level constraints

---

## Architecture

```
User Input (Gradio UI)
        ↓
LangGraph Workflow
        ↓
[Info Collector Node]
        ↓
[Analyzer Node (BMI/BMR/TDEE + USDA)]
        ↓
[Research Node (PubMed + LLM analysis)]
        ↓
[Report Generator Node (LLM)]
        ↓
[PDF Generator Node (ReportLab)]
        ↓
Final Output (Chat + PDF Download)
```

---

## Technologies Used

| Component       | Technology                                        |
| --------------- | ------------------------------------------------- |
| LLM             | OpenAI-compatible Chat API (DeepSeek / GPT-style) |
| Workflow Engine | LangGraph                                         |
| UI              | Gradio                                            |
| Nutrition Data  | USDA FoodData Central API                         |
| Research Data   | PubMed (NCBI E-utilities)                         |
| PDF Generation  | ReportLab                                         |
| Backend         | Python                                            |

---

## Installation

```bash
pip install gradio langchain-openai langgraph requests reportlab
```

---

## Environment Variables

Create a `.env` file or set system variables:

```bash
FDC_API_KEY=your_usda_api_key
LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.gapapi.com/v1
LLM_MODEL=deepseek-chat

NCBI_API_KEY=your_ncbi_api_key
NCBI_EMAIL=your_email@example.com
```

---

## Run the Application

```bash
python app.py
```

Then open the Gradio link shown in terminal.

---

## Key Design Principles

### 1. Single Source of Truth

* The `final_report` is the only source used for both:

  * Chat output
  * PDF generation

### 2. Deterministic PDF Output

* No LLM involvement in PDF creation
* Ensures reproducibility

### 3. Modular LangGraph Pipeline

Each function is a node:

* `info_collector_node`
* `analyzer_node`
* `researcher_node`
* `reporter_node`
* `pdf_generator_node`

---

## PubMed Integration Logic

* Query format:

  ```
  (goal) AND (condition) AND nutrition lifestyle
  ```
* Returns top 3 studies
* Extracted via:

  * esearch (IDs)
  * efetch (XML details)

---

## Guardrails Philosophy

The system is designed to be:

* Educational, not clinical
* Evidence-aware
* Safety-first
* Non-restrictive for health-sensitive users

---

## Output Example Structure

```
AI NUTRITION COACH REPORT

1. User Profile Summary
2. Body Metrics (BMI, BMR, TDEE)
3. Meal Analysis
4. Personalized Nutrition Insights
5. Safety Disclaimer

PUBMED ARTICLE ANALYSIS
- Article 1
- Article 2
- Article 3
```

---

## Limitations

* USDA API may fail for unknown foods
* PubMed abstracts may be incomplete
* LLM output may require validation for strict JSON parsing
* Not a medical device

---

## License

For educational and research use only.


