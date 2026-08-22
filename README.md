# Q&A Evaluator (Streamlit)

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this folder (optional — you can also paste the key
into the sidebar at runtime):

```
GROQ_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

## How it works

1. **Baseline Q&A** — upload a PDF/DOCX containing the reference question and
   answer, and click "Extract baseline Q&A from file" (uses the LLM to pull
   out the question/answer), or type the baseline in manually. You can edit
   the extracted text before running the evaluation.
2. **Individual answers** — upload any number of PDF/DOCX files, one per
   candidate.
3. **Run Evaluation** — for each file, the app extracts the candidate's
   question/answer via the LLM, then scores it (0–100) against the baseline
   answer with a rationale.
4. **Results** — sortable table, top 2 / bottom 2 highlights, and a CSV
   download.
