import os
import openai
import asyncio
import pickle

from dotenv import load_dotenv
# from pinecone import Pinecone, PodSpec

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import UnstructuredReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
# from llama_index.vector_stores.pinecone import PineconeVectorStore

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding

from chromadb.config import Settings
import chromadb


from llama_index.core.extractors import (
    TitleExtractor,
    # QuestionsAnsweredExtractor,
    # SummaryExtractor,
    # KeywordExtractor,
)
from llama_parse import LlamaParse

load_dotenv()
# openai.api_key = os.environ.get("OPENAI_API_KEY")
# pinecone_api_key = os.environ.get("PINECONE_API_KEY")
llama_parse_api_key = os.environ.get("LLAMA_PARSE_API_KEY")

# MODEL = os.getenv("MODEL", "gpt-4-0125-preview")
EMBEDDING = "hkunlp/instructor-xl"

def create_chroma_vector_store(path):
    chroma_client = chromadb.PersistentClient("./chroma_db")
    # chroma_client.delete_collection(name="quickstart")
    chroma_collection = chroma_client.create_collection("test")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def load_chroma_vector_store():
    chroma_client = chromadb.PersistentClient("./chroma_db")
    chroma_collection = chroma_client.get_collection("test")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def get_documents(input_dir):

    parsing_instruction = """Do not parse images"""
    llama_parser = LlamaParse(
        api_key=llama_parse_api_key, result_type="markdown", verbose=True, parsing_instruction=parsing_instruction

    )

    file_extractor = {
        ".pdf": llama_parser,
        # ".html": UnstructuredReader(),
        # ".txt": UnstructuredReader(),
    }
    print("Reading directory")
    director_reader = SimpleDirectoryReader(
        input_dir=input_dir, file_extractor=file_extractor
    )
    print("Starting document reading")
    documents = director_reader.load_data(show_progress=True)
    return documents


def run_pipeline(documents, vector_store, llm, num_workers):
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=32),
            # [TODO] TitleExtractor(llm=llm, num_workers=num_workers),
            # QuestionsAnsweredExtractor(questions=3, llm=llm, num_workers=num_workers),
            # SummaryExtractor(
            #    summaries=["prev", "self"], llm=llm, num_workers=num_workers
            # ),
            # KeywordExtractor(keywords=5, llm=llm, num_workers=num_workers),
            # HuggingFaceEmbedding(model_name=EMBEDDING),
            InstructorEmbedding(model_name=EMBEDDING),
        ],
        vector_store=vector_store,
    )
    for doc in documents:  # Small patch to remove last_accessed_date from metadata
        k = vars(doc)
        del k["metadata"]["last_modified_date"]
    nodes = pipeline.run(documents=documents, show_progress=True, num_workers=num_workers)

    with open("nodes.pickle", 'wb') as f:
        pickle.dump(nodes, f)


def main():
    print("Starting ingestion")
    input_dir = "./data/"
    index_name = "rag-index"
    num_cores = os.cpu_count()
    # num_workers = min(4, num_cores)
    num_workers = 1

    # llm = OpenAI(temperature=0.1, model=MODEL, max_tokens=1024)
    # llm = OpenAILike(
    #     model="llama3-8b-instruct", 
    #     api_base="http://vllm:8000/v1", 
    #     api_key="fake",
    #     temperature=0.0,
    # )
    llm = None

    vector_store = create_chroma_vector_store("./chroma_db")
    documents = get_documents(input_dir)
    print("Starting ingestion pipeline")
    run_pipeline(documents, vector_store, llm, num_workers)


if __name__ == "__main__":
    main()
