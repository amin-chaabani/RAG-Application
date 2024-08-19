from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from annoy import AnnoyIndex
import gradio as gr
from pymongo import MongoClient
import boto3
import redis
import os

# Redis connection setup
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

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

# Define LLM
llm = Ollama(model="llama3")

# Define the prompt template
prompt = """
1. Utilisez les éléments de contexte suivants pour répondre à la question à la fin.
2. Si vous ne connaissez pas la réponse, dites simplement "Je ne sais pas" mais n'inventez pas de réponse.\n
3. Gardez la réponse concise et limitée à 3 ou 4 phrases.

Contexte: {context}

Question: {question}

Réponse utile:"""

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

# Define the embedder and Annoy index globally
embedder = HuggingFaceEmbeddings()
annoy_index = None
all_documents = []


# Define a User model
def create_user(user_data):
    user_id = users_collection.insert_one(user_data).inserted_id
    return user_id


# Define a Vector Store model
def create_vector_store(user_id, s3_key):
    vector_store_id = vector_store_collection.insert_one({
        'user_id': user_id,
        's3_path': s3_key
    }).inserted_id
    return vector_store_id


# Function to process PDFs and build the Annoy index
def process_pdfs(files):
    global annoy_index, all_documents
    user_id = create_user({"username": "default_user"})  # Replace with actual user logic

    if not files:
        return "No PDF files were uploaded."

    all_documents = []
    for file in files:
        # Load the PDF
        pdf_loader = PDFPlumberLoader(file.name)
        docs = pdf_loader.load()

        # Split the documents
        text_splitter = SemanticChunker(embedder)
        chunks = text_splitter.split_documents(docs)
        all_documents.extend(chunks)

    # Calculate vector dimension using a sample embedding
    sample_vector = embedder.embed_query("sample text")
    vector_dimension = len(sample_vector)

    # Initialize Annoy index
    annoy_index = AnnoyIndex(vector_dimension, 'angular')

    # Add documents to the Annoy index
    for i, doc in enumerate(all_documents):
        vector = embedder.embed_query(doc.page_content)
        annoy_index.add_item(i, vector)

    # Build the index
    annoy_index.build(10)

    # Save the Annoy index to a file
    index_filename = f"{user_id}_annoy_index.ann"

    try:
        annoy_index.save(index_filename)

        # Upload the index file to MinIO
        with open(index_filename, 'rb') as f:
            s3_key = f"{user_id}/{index_filename}"
            s3_client.put_object(Bucket='my-minio-bucke', Key=s3_key, Body=f)

        # Store the S3 key in MongoDB
        create_vector_store(user_id, s3_key)

    finally:
        # Ensure the local file is removed after upload
        if os.path.exists(index_filename):
            try:
                os.remove(index_filename)
            except PermissionError as e:
                print(f"Failed to delete {index_filename}: {e}")

    return f"PDF files have been successfully loaded and processed for user {user_id}."


# Custom retriever based on Annoy index
class AnnoyRetriever:
    def __init__(self, index, documents):
        self.index = index
        self.documents = documents

    def get_relevant_documents(self, query):
        query_vector = embedder.embed_query(query)
        nearest_neighbors = self.index.get_nns_by_vector(query_vector, 3)
        return [self.documents[i] for i in nearest_neighbors]


# Function to handle chat interaction
def respond(message, history):
    global annoy_index, all_documents
    if history is None:
        history = []

    print(f"Message reçu : {message}")

    cached_response = redis_client.get(message)
    if cached_response:
        print(f"Réponse en cache trouvée pour '{message}': {cached_response}")
        return [("Quick response (cache): " + cached_response, "")], history

    print("Aucune réponse en cache trouvée, traitement de la requête...")

    if annoy_index is None:
        return [("Please upload PDF files first.", "")], history

    retriever = AnnoyRetriever(annoy_index, all_documents)
    retrieved_documents = retriever.get_relevant_documents(message)

    context = "\n".join([doc.page_content for doc in retrieved_documents])

    response = llm_chain.run({"context": context, "question": message})
    history.append((message, response))

    # Stocker la réponse dans Redis
    redis_client.setex(message, 3600, response)
    print(f"Réponse stockée dans Redis pour '{message}': {response}")

    return history, history


# Gradio interface for PDF upload
file_interface = gr.Interface(
    fn=process_pdfs,
    inputs=gr.File(file_count="multiple", type="filepath"),
    outputs="text",
    title="Charger des fichiers PDF"
)

# Gradio interface for chatbot interaction
chat_interface = gr.Interface(
    fn=respond,
    inputs=[gr.Textbox(placeholder="Posez une question liée aux documents chargés"), gr.State()],
    outputs=[gr.Chatbot(), gr.State()],
    title="RAG MBA Chatbot"
)

# Combine both interfaces in a tabbed interface
gr.TabbedInterface([file_interface, chat_interface], ["Charger des fichiers PDF", "Chatbot"]).launch(share=True)
