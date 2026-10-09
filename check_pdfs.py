import glob
import pymupdf

files = glob.glob("data/docs/*.pdf")
if not files:
    print("No PDFs found in data/docs")

for f in files:
    doc = pymupdf.open(f)
    chars = sum(len(p.get_text().strip()) for p in doc)
    print(f, "-", len(doc), "pages,", chars, "characters")