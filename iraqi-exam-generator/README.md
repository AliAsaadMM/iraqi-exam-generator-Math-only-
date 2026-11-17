# Iraqi Exam Sheet Generator (صانع الأسئلة الامتحانية العراقية)

A Streamlit web app that converts handwritten exam photos or textbook PDFs/images into formatted, print-ready Iraqi Ministry of Education exam PDFs with full Arabic RTL and LaTeX math support.

## Features
- **Handwritten OCR**: Extracts and sequentially numbers questions from photos using Gemini AI.
- **Textbook Question Curation**: Selects a custom number of exam questions from PDF chapters or book page images.
- **LaTeX Math Rendering**: Automatically converts `$ ... $` and `$$ ... $$` equations into crisp vector SVGs.
- **Official Layout & Auto-Fit**: Formats headers/footers in traditional Iraqi exam style and scales typography to fit 1 or 2 pages on demand.

## Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt