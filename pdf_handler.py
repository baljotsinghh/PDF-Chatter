import fitz  # PyMuPDF

class PDFTextExtractor:
    def __init__(self, file):
        self.file = file
        self.doc = fitz.open(stream=file.read(), filetype="pdf")
    
    def extract_text(self) -> str:
        text = ""
        for page in self.doc:
            text += page.get_text()
        return text