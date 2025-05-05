from IAModel.model import DocumentClassifier, Model
from transformers.models.bert import BertTokenizer, BertForSequenceClassification
import torch
import pdfplumber


class DocBERTModel(Model, DocumentClassifier):

    def __init__(self, name, path, description, load_params, run_params) -> None:
        super().__init__(name, path, description)
        self.name = name
        self.path = path
        self.load_params = load_params
        self.run_params = run_params
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        """
        Loads the BERT model and tokenizer into memory.
        """
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.path)
            self.model = BertForSequenceClassification.from_pretrained(
                self.path, **self.load_params
            )
            self.model.to(self.device)
            print(f"Model '{self.name}' loaded successfully from '{self.path}'")
        except Exception as e:
            print(f"Failed to load model '{self.name}': {e}")
            raise

    def unload(self):
        """
        Unloads the BERT model from memory.
        """
        if self.model:
            self.model = None
            self.tokenizer = None
            print(f"Model '{self.name}' unloaded from memory.")

    def run(self, text):
        """
        Runs the BERT model on input text and returns classification logits.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError(
                f"Model '{self.name}' is not loaded. Please load the model first."
            )

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
            return predicted_class
        except Exception as e:
            print(f"Error running model '{self.name}': {e}")
            raise

    def classify(self, pdf_path):
        """
        Classifies the content of a PDF document using the model.
        """
        if not self.model:
            raise RuntimeError(
                f"Model '{self.name}' is not loaded. Please load the model first."
            )

        pdf_text = self._extract_text_from_pdf(pdf_path)
        if not pdf_text:
            raise ValueError(f"No text extracted from PDF: {pdf_path}")

        return self.run(pdf_text)

    def _extract_text_from_pdf(self, pdf_path):
        """
        Helper function to extract text from a PDF file using pdfplumber.
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"Error extracting text from PDF '{pdf_path}': {e}")
            raise
