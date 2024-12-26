import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class Embedder:
    def __init__(self, file_name, db_name="db"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filename = file_name
        self.file_path = os.path.join(current_dir, "documents", file_name)
        self.db_dir = os.path.join(current_dir, db_name)
        self._check_file_exists()

    def _check_file_exists(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The file {self.file_path} does not exist. Please check the path."
            )

    def load_text(self):
        loader = TextLoader(self.file_path, 'utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs

    def create_vector_store(self):
        huggingface_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        persistent_directory = os.path.join(self.db_dir, self.filename)
        if not os.path.exists(persistent_directory):
            Chroma.from_documents(
                self.load_text(), huggingface_embeddings, persist_directory=persistent_directory)
    
    def chat(self,query):
        embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        persistent_directory = os.path.join(self.db_dir, self.filename)
        db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
            )
        relevant_docs = retriever.invoke(query)

        print("\n--- Relevant Documents ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")   
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'. And I want precise answers only. don't add any extra information."
        )
        model = ChatOllama(
            model="llama3.2",
            temperature=0,
            # other params...
            )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=combined_input),
            ]
        result = model.invoke(messages)
        return result.content