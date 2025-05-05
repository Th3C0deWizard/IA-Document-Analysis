from IAModel.model import DocumentClassifier, DocumentExtractor, Model
from llama_cpp import Llama
import pdfplumber


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
        prompt = f"""<|begin_of_text|><|user|>
        You are an expert document analyzer. Your task is to classify the document content provided below. Respond with labels separated by commas.

        Instructions:
        - Do not include any explanation.
        - Use concise words.
        - respond with labels only.

        Document content:
        \"\"\"{pdf_text}\"\"\"
        <|end_of_text|><|assistant|>"""
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
        prompt = f"""<|begin_of_text|><|user|>
        You are an expert document analyzer. Your task is to extract the key information from the document content provided below.

        Instructions:
        - Do not include any explanation.
        - Only respond with a JSON file. 

        Document content:
        \"\"\"{pdf_text}\"\"\"
        <|end_of_text|><|assistant|>"""

        extraction = self.run(prompt)
        return extraction.strip()

    def _extract_text_from_pdf(self, pdf_path):
        """
        Helper function to extract text from a PDF file using pdfplumber.
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""  # Handle empty pages gracefully
            return text
        except Exception as e:
            print(f"Error extracting text from PDF '{pdf_path}': {e}")
            raise

    def _extract_text_and_boxes_from_pdf(self, pdf_path):
        """
        Helper function to extract text and bounding boxes from a PDF file using pdfplumber.
        """
        try:
            text = []
            quads = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    for word in page.extract_words():
                        text.append(word["text"])
                        quads.append(
                            [
                                word["x0"],
                                word["top"],
                                word["x1"],
                                word["top"],
                                word["x1"],
                                word["bottom"],
                                word["x0"],
                                word["bottom"],
                            ]
                        )
            return {
                "text": text,
                "quads": quads,
                "width": pdf.pages[0].width,
                "height": pdf.pages[0].height,
            }
        except Exception as e:
            print(f"Error extracting text and boxes from PDF '{pdf_path}': {e}")
            raise
