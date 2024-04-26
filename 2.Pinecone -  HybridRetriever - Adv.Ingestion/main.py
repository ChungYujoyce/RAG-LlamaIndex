import os
import chainlit as cl

from llama_index.core import Settings, VectorStoreIndex,SimpleDirectoryReader, StorageContext, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# openai.api_key = os.environ.get("OPENAI_API_KEY")
# pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# MODEL = os.getenv("MODEL", "gpt-4-0125-preview")
# EMBEDDING = os.getenv("EMBEDDING", "text-embedding-3-large")
SYS_PROMPT = PromptTemplate(
    """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

As a helpful assistant, you will utilize the provided document to answer user questions. 
Read the given document before providing answers and think step by step. 
The document has an order of paragraphs with a higher correlation to the questions from the top to the bottom. 
The answer may be hidden in the tables, so please find it as closely as possible. 
Do not use any other information to answer the user. Provide a etailed answer to the question.
Also, please provide the answer in the following order of priorities if applicable:
Firstly, emphasize GPU characteristics and GPU products.
Secondly, Give prominence to power-related specifications such as fan cooling or liquid cooling, power consumption, and so on.
Thirdly, If applicable, mention green computing.
Remember, please don't provide any fabricated information, ensuring that everything stated is accurate and true.

Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.<|eot_id|>
{query_str}\
"""
)
CONVERSATION_PROMPT = "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
RESPONSE_PROMPT = "<|start_header_id|>{role}<|end_header_id|>\n\n"
EMBEDDING = os.getenv("EMBEDDING", "hkunlp/instructor-xl")

def load_chroma_vector_store():
    chroma_client = chromadb.PersistentClient("./chroma_db")
    chroma_collection = chroma_client.get_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store

@cl.cache
def load_context():
    Settings.llm = OpenAILike(
        model="llama3-8b-instruct", 
        api_base="http://vllm:8000/v1", 
        api_key="fake",
        temperature=0.0,
        max_tokens=256,
    )
    Settings.embed_model = InstructorEmbedding(model_name="hkunlp/instructor-xl")
    Settings.num_output = 1024
    Settings.context_window = 8192
    
    vector_store = load_chroma_vector_store()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )
    return index


@cl.on_chat_start
async def start():
    index = load_context()

    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=4,
        text_qa_template=SYS_PROMPT,
        vector_store_query_mode="hybrid",
    )
    cl.user_session.set("query_engine", query_engine)

    message_history = []
    cl.user_session.set("message_history", message_history)

    await cl.Message(
        author="assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


async def set_sources(response, response_message):
    label_list = []
    count = 1
    for sr in response.source_nodes:
        elements = [
            cl.Text(
                name="S" + str(count),
                content=f"{sr.node.text}",
                display="side",
                size="small",
            )
        ]
        response_message.elements = elements
        label_list.append("S" + str(count))
        await response_message.update()
        count += 1
    response_message.content += "\n\nSources: " + ", ".join(label_list)
    await response_message.update()


@cl.on_message
async def main(user_message: cl.Message):
    n_history_messages = 4
    query_engine = cl.user_session.get("query_engine")
    message_history = cl.user_session.get("message_history")
    prompt_template = ""
    print(message_history)
    for past_message in message_history:
        prompt_template += CONVERSATION_PROMPT.format(
            role=past_message['author'],
            content=past_message['content'],
        )

    prompt_template += CONVERSATION_PROMPT.format(
        role="user",
        content=user_message.content,
    ) 
    
    prompt_template += RESPONSE_PROMPT.format(
        role='assistant',
    ) 

    response = await cl.make_async(query_engine.query)(prompt_template)
    
    assistant_message = cl.Message(content="", author="assistant")
    for token in response.response_gen:
        await assistant_message.stream_token(token)
    if response.response_txt:
        assistant_message.content = response.response_txt
    await assistant_message.send()

    message_history.append({"author": "user", "content": user_message.content})
    message_history.append({"author": "assistant", "content": assistant_message.content})
    message_history = message_history[-n_history_messages:]
    cl.user_session.set("message_history", message_history)

    if response.source_nodes:
        await set_sources(response, assistant_message)
