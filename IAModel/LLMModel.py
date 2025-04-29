from IAModel.model import DocumentClassifier, DocumentExtractor, Model
from llama_cpp import Llama
import fitz  # PyMuPDF for PDF parsing


class LLMModel(Model, DocumentClassifier, DocumentExtractor):

    def __init__(self, name, path, description, load_params, run_params) -> None:
        super().__init__(name, path, description)
        self.name = name
        self.path = path
        self.load_params = load_params
        self.run_params = run_params
        self.model = None  # Placeholder for the LLaMA model instance

    def load(self):
        """
        Loads the LLaMA model into memory using the given path and parameters.
        """
        try:
            self.model = Llama(model_path=self.path, **self.load_params)
            print(f"Model '{self.name}' loaded successfully from '{self.path}'")
        except Exception as e:
            print(f"Failed to load model '{self.name}': {e}")
            raise

    def unload(self):
        """
        Unloads the LLaMA model from memory.
        """
        if self.model:
            self.model = None
            print(f"Model '{self.name}' unloaded from memory.")

    def run(self, prompt):
        """
        Runs the LLaMA model with the given prompt and returns the output.
        """
        if not self.model:
            raise RuntimeError(
                f"Model '{self.name}' is not loaded. Please load the model first."
            )

        try:
            result = self.model(prompt, **self.run_params)
            return result["choices"][0]["text"]
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

        # Extract text from the PDF
        pdf_text = self._extract_text_from_pdf(pdf_path)
        if not pdf_text:
            raise ValueError(f"No text extracted from PDF: {pdf_path}")

        # Create a classification prompt
        prompt = f"Classify the following document providing just the correspoding labels (words), example (billing, clinical history, scientific article, e.t.c):\n\n{pdf_text[:1000]}"
        classification = self.run(prompt)
        return classification.strip()

    def extract(self, pdf_path):
        """
        Extracts information from a PDF document using the model.
        """
        if not self.model:
            raise RuntimeError(
                f"Model '{self.name}' is not loaded. Please load the model first."
            )

        # Extract text from the PDF
        pdf_text = self._extract_text_from_pdf(pdf_path)
        if not pdf_text:
            raise ValueError(f"No text extracted from PDF: {pdf_path}")

        # Create an extraction prompt
        prompt = f"Extract the relevant information from the following document and provide it in JSON format:\n\n{pdf_text[:1000]}"
        extraction = self.run(prompt)
        return extraction.strip()

    def _extract_text_from_pdf(self, pdf_path):
        """
        Helper function to extract text from a PDF file using PyMuPDF (fitz).
        """
        try:
            with fitz.open(pdf_path) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF '{pdf_path}': {e}")
            raise
