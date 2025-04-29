from pydantic import InstanceOf


class Model:

    def __init__(self, name, path, description) -> None:
        self.name = name
        self.path = path
        self.description = description

    def load(self):
        pass

    def unload(self):
        pass

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "classifier": isinstance(self, DocumentClassifier),
            "extractor": isinstance(self, DocumentExtractor),
        }


"""
Interface for models that are capable of document classification task
"""


class DocumentClassifier:

    def classify(self, pdf_path):
        pass


"""
Interface for models that are capable of document key information extraction task
"""


class DocumentExtractor:

    def extract(self, pdf_path):
        pass
