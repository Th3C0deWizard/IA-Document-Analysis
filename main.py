from contextlib import asynccontextmanager
from IAModel.LLMModel import LLMModel
from IAModel.DocBert import DocBERTModel
from typing import Optional
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import shutil


class ModelManager:

    def __init__(self):
        self.MODELS_CONFIG_PATH = os.getenv("MODELS_CONFIG_PATH", "config/models/")
        self.models = {}
        self.docClassifiers = {}
        self.docExtractors = {}
        self.loaded_model = None

    def get_llms_from_dir(self):
        """
        Reads JSON files from the specified directory, parses them, and constructs LLMModel objects.
        """
        project_root = os.path.abspath(os.path.dirname(__file__))
        directory = f"{project_root}/{self.MODELS_CONFIG_PATH}/llms"
        try:
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, "r") as file:
                        data = json.load(file)
                        # Extract required attributes for the LLMModel
                        name = data.get("name")
                        path = data.get("path")
                        description = data.get("description")
                        load_params = data.get("load_params")
                        run_params = data.get("run_params")
                        if not name and not path:
                            raise ValueError(
                                f"Invalid JSON file {filename}: 'name' and 'path' are required."
                            )

                        # Create an instance of LLMModel
                        model = LLMModel(
                            name=name,
                            path=f"{project_root}/resources/models/llms/{path}",
                            load_params=load_params,
                            run_params=run_params,
                            description=description,
                        )
                        self.models[name] = model
                        self.docClassifiers[name] = model
                        self.docExtractors[name] = model

        except Exception as e:
            print(f"Error initializing models: {e}")

    def initialize(self):
        print("Initializing model manager...")
        self.get_llms_from_dir()

        # instanciate docbert Model
        project_root = os.path.abspath(os.path.dirname(__file__))
        docbert_config_path = os.path.join(
            project_root, self.MODELS_CONFIG_PATH, "docbert.json"
        )

        try:
            with open(docbert_config_path, "r") as file:
                data = json.load(file)
                name = data.get("name")
                path = data.get("path")
                description = data.get("description")
                load_params = data.get("load_params")
                run_params = data.get("run_params")
                if not name or not path:
                    raise ValueError("DocBERT config must include 'name' and 'path'")
                model = DocBERTModel(
                    name=name,
                    path=os.path.join(project_root, "resources/models/", path),
                    load_params=load_params,
                    run_params=run_params,
                    description=description,
                )
                self.models[name] = model
                self.docClassifiers[name] = model
        except Exception as e:
            print(f"Error initializing DocBERT model: {e}")

    def load_model(self, model_name):
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")

        if self.loaded_model == model_name:
            print(f"Model '{model_name}' is already loaded.")
            return

        if self.loaded_model:
            self.unload_model()

        model.load()
        self.loaded_model = model_name

    def unload_model(self):
        model = self.models.get(self.loaded_model)
        if not model:
            raise ValueError(f"There is no model loaded")
        model.unload()
        self.loaded_model = None

    def get_loaded_model(self):
        return self.models.get(self.loaded_model)

    def is_classifier(self, model_name):
        return self.docClassifiers.get(model_name) is not None

    def is_extractor(self, model_name):
        return self.docExtractors.get(model_name) is not None


# Crear una función de ciclo de vida con lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar el estado global
    model_manager = ModelManager()
    model_manager.initialize()

    # Proveer el estado global al contexto de la aplicación
    yield {"model_manager": model_manager}
    print("Aplicación cerrándose. Limpieza si es necesaria.")
    model_manager.unload_model()


# Crear la instancia de FastAPI con lifespan
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint para listar los modelos disponibles con filtro por tarea
@app.get("/api/v1/models/")
async def list_models(
    request: Request,
    task: Optional[str] = Query(
        None, description="Filter models by task: 'classification' or 'extraction'"
    ),
):
    model_manager: ModelManager = request.state.model_manager
    if task == "classification":
        return {
            "models": list(
                [model.to_dict() for model in model_manager.docClassifiers.values()]
            )
        }
    elif task == "extraction":
        return {
            "models": list(
                [model.to_dict() for model in model_manager.docExtractors.values()]
            )
        }
    else:
        return {
            "models": list([model.to_dict() for model in model_manager.models.values()])
        }


@app.post("/api/v1/models/{model_name}/{task}")
async def predict_classify(
    request: Request,
    model_name: str,
    task: str,
    file: UploadFile = File(...),
):
    model_manager: ModelManager = request.state.model_manager
    model = model_manager.models.get(model_name)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if task not in ["classification", "extraction"]:
        raise HTTPException(status_code=400, detail="Invalid task specified")
    if task == "classification" and not model_manager.is_classifier(model_name):
        raise HTTPException(status_code=400, detail="Model is not a classifier")
    if task == "extraction" and not model_manager.is_extractor(model_name):
        raise HTTPException(status_code=400, detail="Model is not an extractor")

    model_manager.load_model(model_name)

    # Save the uploaded file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = (
            model.classify(temp_file_path)
            if task == "classification"
            else model.extract(temp_file_path)
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
