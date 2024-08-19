from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import gradio as gr
from pymongo import MongoClient
import boto3
import redis
from annoy import AnnoyIndex

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# LLM Setup
llm = Ollama(model="llama3")

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['ragdb']
users_collection = db['users']
vector_store_collection = db['vector_stores']

# MinIO setup
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='amine',
    aws_secret_access_key='amine123',
    region_name='us-east-1',
    config=boto3.session.Config(signature_version='s3v4')
)

# Define a User model
def create_user(user_data):
    user_id = users_collection.insert_one(user_data).inserted_id
    return user_id

# Define a Vector Store model
def create_vector_store(user_id, vector_data):
    vector_store_id = vector_store_collection.insert_one({
        'user_id': user_id,
        'vector_data': vector_data,
        's3_path': None
    }).inserted_id
    return vector_store_id

def upload_to_minio(user_id, vector_store_id, vector_data):
    s3_key = f"{user_id}/{vector_store_id}.ann"
    s3_client.put_object(Bucket='my-minio-bucket', Key=s3_key, Body=vector_data)
    vector_store_collection.update_one({'_id': vector_store_id}, {'$set': {'s3_path': s3_key}})

# Setup prompts and chains
prompt = """
1. Utilize the following context to answer the question at the end.
2. If you do not know the answer, simply say "I do not know" but do not make up an answer.\n
3. Keep the answer concise and limited to 3 or 4 sentences.

Context: {context}

Question: {question}

Useful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

llm_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    callbacks=None,
    verbose=True
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None
)

embedder = HuggingFaceEmbeddings()

def load_pdfs(files, user_id):
    if not files:
        return "No PDF files were uploaded."
    all_documents = []
    for file in files:
        loader = PDFPlumberLoader(file.name)
        docs = loader.load()
        text_splitter = SemanticChunker(embedder)
        documents = text_splitter.split_documents(docs)
        all_documents.extend(documents)

    vector_dimension = embedder.vector_size
    annoy_index = AnnoyIndex(vector_dimension, 'angular')

    for i, doc in enumerate(all_documents):
        vector = embedder.embed_query(doc.page_content)
        annoy_index.add_item(i, vector)

    annoy_index.build(10)

    vector_store_id = create_vector_store(user_id, annoy_index)
    annoy_index.save('annoy_index.ann')
    upload_to_minio(user_id, vector_store_id, open('annoy_index.ann', 'rb').read())

    return f"PDF files have been successfully loaded and processed for user {user_id}."

def delete_vector_store(user_id, vector_store_id):
    vector_store = vector_store_collection.find_one({'_id': vector_store_id, 'user_id': user_id})
    if vector_store:
        s3_client.delete_object(Bucket='my-minio-bucket', Key=vector_store['s3_path'])
        vector_store_collection.delete_one({'_id': vector_store_id})
        return f"Vector store {vector_store_id} for user {user_id} has been deleted."
    return "Vector store not found or does not belong to the user."

def respond(message, history, user_id):
    if history is None:
        history = []
    cached_response = redis_client.get(message)
    if cached_response:
        return [("Quick response (cache): " + cached_response, "")], history
    vector_store = vector_store_collection.find_one({'user_id': user_id})
    if not vector_store:
        return [("Please upload PDF files first.", "")], history
    # Implement retrieval logic using Annoy here
    response = "This is a mock response"  # Placeholder response
    history.append((message, response))
    redis_client.setex(message, 3600, response)
    return history, history

file_interface = gr.Interface(
    fn=load_pdfs,
    inputs=gr.File(file_count="multiple", type="filepath"),
    outputs="text",
    title="Upload PDF Files"
)

chat_interface = gr.Interface(
    fn=respond,
    inputs=[gr.Textbox(placeholder="Ask a question related to plant diseases"), gr.State()],
    outputs=[gr.Chatbot(), gr.State()],
    title="RAG MBA Chatbot",
    examples=["What are the differences between LLAMA3:8B and LLAMA3:70B?", "What is llama?"],
    cache_examples=False
)

gr.TabbedInterface([file_interface, chat_interface], ["Upload PDF Files", "Chatbot"]).launch(
    share=True,
    server_name='0.0.0.0',
    auth=[("demo", "mba91")],
    server_port=7773
)
