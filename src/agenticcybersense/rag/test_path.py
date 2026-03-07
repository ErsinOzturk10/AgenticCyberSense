from pathlib import Path

DATA_PATH = Path("/Users/merveatay/Projekte/AgenticAI/AgenticCyberSense.v2/src/agenticcybersense/data")

print("DATA_PATH:", DATA_PATH)

pdfs = list(DATA_PATH.glob("*.pdf"))

print("PDFS:", pdfs)
print("COUNT:", len(pdfs))