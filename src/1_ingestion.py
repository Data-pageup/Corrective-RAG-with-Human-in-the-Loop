from pathlib import Path
import json
import fitz  #PyMuPDF

# Project_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents = True, exist_ok = True)


def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF."""

    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start= 1):

            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append({
                "source": pdf_path.name,
                "page":page_number,
                "text": text
            })
    return pages

def ingest_pdfs():
        """Extract text from all PDFs in the raw directory."""

        pdf_files = list(RAW_DIR.glob("*.pdf"))

        if not pdf_files:
             print(f"No pdf's found in {RAW_DIR}")
             return
        all_pages = []

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")

            try:
                pages = extract_pdf(pdf_path)
                all_pages.extend(pages)
                print(f"Extracted {len(pages)} pages")

            except Exception as error:
                 print(f"failed to process {pdf_path.name}:{error}")


        output_path = PROCESSED_DIR / "extracted_pages.jsonl"

        with output_path.open("w", encoding="utf-8") as file:
            for page in all_pages:
                file.write(
                    json.dumps(page, ensure_ascii=False) + "\n"
                )

        print(f"\nTotal extracted pages: {len(all_pages)}")
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    ingest_pdfs()

