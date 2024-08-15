from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import gradio as gr

# Définir LLM
llm = Ollama(model="llama3")

# Définir le prompt
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

# Définir l'embedder et le vecteur
embedder = HuggingFaceEmbeddings()
vector = None


def load_pdfs(files):
    global vector
    if not files:
        return "Aucun fichier PDF n'a été téléchargé."

    all_documents = []
    for file in files:
        loader = PDFPlumberLoader(file.name)
        docs = loader.load()
        text_splitter = SemanticChunker(HuggingFaceEmbeddings())
        documents = text_splitter.split_documents(docs)
        all_documents.extend(documents)

    vector = FAISS.from_documents(all_documents, embedder)
    return "Les fichiers PDF ont été chargés et traités avec succès."


def respond(message, history):
    if history is None:
        history = []
    if vector is None:
        return [("Veuillez d'abord charger les fichiers PDF.", "")], history
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True
    )
    response = qa(message)["result"]
    history.append((message, response))
    return history, history


# Interface Gradio
file_interface = gr.Interface(
    fn=load_pdfs,
    inputs=gr.File(file_count="multiple", type="filepath"),  # Corrected type
    outputs="text",
    title="Charger des fichiers PDF"
)

chat_interface = gr.Interface(
    fn=respond,
    inputs=[gr.Textbox(placeholder="Posez une question liée aux plantes et à leurs maladies"), gr.State()],
    outputs=[gr.Chatbot(), gr.State()],
    title="RAG MBA chatBOT",
    examples=["Quelles sont les différences entre LLAMA3:8B et LLAMA3:70B ?", "Qu'est-ce que llama ?"],
    cache_examples=False  # Mise à jour pour désactiver le cache des exemples
)

gr.TabbedInterface([file_interface, chat_interface], ["Charger des fichiers PDF", "Chatbot"]).launch(share=True,
                                                                                                     server_name='0.0.0.0',
                                                                                                     auth=[("demo",
                                                                                                            "mba91")],
                                                                                                     server_port=7773)
