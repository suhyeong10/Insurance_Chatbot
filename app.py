import os
import pickle
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import LongContextReorder

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, FewShotPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


llm = ChatOllama(
    model='codingchild/insurance-gemma-3-12b',
    base_url="http://localhost:11434",
    temperature=0.7,
    top_k=40,
    top_p=0.92,
    num_predict=1024,
    repeat_penalty=1.2,
    num_gpu=1,
    num_ctx=2048,
    num_thread=8,
    disable_streaming=False
)

SYSTEM_PREFIX = """You are a knowledgeable and courteous insurance assistant.
Your role is to help users understand life insurance policies, evaluate their needs,
and suggest appropriate coverage options. Speak in a respectful and supportive tone,
avoiding sales pressure while offering clear, accurate information.

You can use only 1024 tokens for your response.

### Context
{context}
"""

system_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PREFIX),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}")
])


class Loader:
    def __init__(self, root_dir, vector_store_path, document_save_path, model_name, embedding_model):
        self.root_dir = root_dir
        self.vector_store_path = vector_store_path
        self.pdf_path = os.path.join(root_dir, "documents")

        if not os.path.exists(os.path.join(root_dir, document_save_path)):
            self.doc_list = self.get_docs()
            self.text_splitters = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=AutoTokenizer.from_pretrained(model_name),
                chunk_size=2048,
                chunk_overlap=256
            )
            docs_splits = self.text_splitters.split_documents(self.doc_list)

            with open(os.path.join(root_dir, document_save_path), 'wb') as f:
                pickle.dump(docs_splits, f)
        else:
            with open(os.path.join(root_dir, document_save_path), 'rb') as f:
                docs_splits = pickle.load(f)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            cache_folder='../../embedding_models',
            model_kwargs={"device": "cuda:0", "model_kwargs": {'torch_dtype': torch.bfloat16}},
            encode_kwargs={"normalize_embeddings": True}
        )

        if os.path.exists(self.vector_store_path):
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS.from_documents(
                documents=docs_splits,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.COSINE
            )
            self.vector_store.save_local(self.vector_store_path)

    def get_docs(self):
        docs = []
        pdf_files = [os.path.join(self.pdf_path, f) for f in os.listdir(self.pdf_path) if f.endswith('.pdf')]
        for file in tqdm(pdf_files, desc="Loading PDF files", unit="file", ncols=150):
            docs.append(PyPDFLoader(file).load())
        return [item for sublist in docs for item in sublist]

    def get_retriever(self, k=3):
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})


reorder = LongContextReorder()
def format_docs(docs):
    reordered = reorder.transform_documents(docs)
    return "\n\n".join([d.page_content for d in reordered])


store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


examples = [
    {
        "question": "I'm 30 and I have lung cancer. Can I still get life insurance?",
        "query": "life insurance with lung cancer"
    },
    {
        "question": "Can I convert my group insurance after leaving my job?",
        "query": "group insurance conversion after job change"
    },
    {
        "question": "What are the differences between term and whole life insurance?",
        "query": "term vs whole life insurance"
    },
    {
        "question": "Can my spouse and I share a single life insurance policy?",
        "query": "joint life insurance for spouses"
    },
    {
        "question": "How does the accelerated death benefit rider work?",
        "query": "accelerated death benefit rider explanation"
    }
]

example_prompt = PromptTemplate.from_template(
    "User input: {question}\nSearch query: {query}"
)

query_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a helpful assistant. Convert the following user input into a short, keyword-based search query for insurance-related documents.",
    suffix="User input: {question}\nSearch query:",
    input_variables=["question"]
)

query_chain = query_prompt | llm | StrOutputParser()


loader = Loader(
    root_dir='database',
    vector_store_path='database/vector_store',
    document_save_path='docs.pkl',
    model_name='codingchild/insurance-gemma-3-12b',
    embedding_model='BAAI/bge-large-en-v1.5',
)

retriever = loader.get_retriever()

rag_chain = (
    {
        "context": (lambda x: {"question": x["input"]}) | query_chain | retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | system_prompt
    | llm
    | StrOutputParser()
)

chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


if __name__ == "__main__":
    while True:
        query = input("User: ")

        if query.lower() == 'exit':
            break

        print('Bot: ', end='', flush=True)
        for chunk in chain.stream(
            {"input": query},
            config={"configurable": {"session_id": "MyTestSessionID"}}
        ):  
            print(chunk, end='', flush=True)
        
        print()
