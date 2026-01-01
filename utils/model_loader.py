import os
import sys
from dotenv import load_dotenv

from utils.config_loader import load_config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

log = CustomLogger().get_logger(__name__)


class ModelLoader:
    """
    Utility class to load embedding models and LLMs.
    """

    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        log.info("Configuration loaded", config_keys=list(self.config.keys()))

    def _validate_env(self):
        required_vars = [
            "GROQ_API_KEY",
            "GOOGLE_API_KEY",
            "HUGGINGFACEHUB_API_TOKEN"
        ]

        self.api_keys = {k: os.getenv(k) for k in required_vars}
        missing = [k for k, v in self.api_keys.items() if not v]

        if missing:
            log.error("Missing environment variables", missing_vars=missing)
            raise DocumentPortalException("Missing environment variables", sys)

        log.info("Environment variables validated")

    # -------------------- EMBEDDINGS --------------------

    def load_embeddings(self):
        """
        Load and return the embedding model.
        """
        try:
            log.info("Loading embedding model")

            cfg = self.config["embedding_model"]
            provider = cfg["provider"]
            model_name = cfg["model_name"]

            if provider != "huggingface":
                raise ValueError(f"Unsupported embedding provider: {provider}")

            embeddings = HuggingFaceEmbeddings(
                model_name=model_name
            )

            log.info("Embedding model loaded successfully", model=model_name)
            return embeddings

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)



    # -------------------- LLM --------------------

    def load_llm(self):
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "google")

        if provider_key not in llm_block:
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        cfg = llm_block[provider_key]
        provider = cfg["provider"]

        log.info("Loading LLM", provider=provider, model=cfg["model_name"])

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=cfg["model_name"],
                google_api_key=self.api_keys["GOOGLE_API_KEY"],
                temperature=cfg.get("temperature", 0),
                max_output_tokens=cfg.get("max_output_tokens", 2048)
            )

        elif provider == "groq":
            return ChatGroq(
                model=cfg["model_name"],
                api_key=self.api_keys["GROQ_API_KEY"],
                temperature=cfg.get("temperature", 0)
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# -------------------- TEST --------------------
if __name__ == "__main__":
    loader = ModelLoader()

     # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")
    
    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")
