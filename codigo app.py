from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA

from langchain_core.language_models.llms import LLM

from langchain_community.embeddings import FakeEmbeddings

from typing import Optional, List


# =========================
# MODELO LLM SIMPLES
# =========================

class SimpleLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "simple-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return f"\nResposta baseada nos documentos encontrados:\n\n{prompt[:500]}"


# =========================
# CARREGAR TEXTO
# =========================

loader = TextLoader(
    "textos/ia.txt",
    encoding="utf-8"
)

documents = loader.load()


# =========================
# DIVIDIR EM CHUNKS
# =========================

text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

docs = text_splitter.split_documents(documents)


# =========================
# EMBEDDINGS SIMPLES
# =========================

embeddings = FakeEmbeddings(size=1352)


# =========================
# VECTOR DATABASE
# =========================

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)


# =========================
# RETRIEVAL QA
# =========================

qa = RetrievalQA.from_chain_type(
    llm=SimpleLLM(),
    retriever=vectorstore.as_retriever()
)


# =========================
# LOOP
# =========================

print("\nSistema VectorDB + RetrievalQA")
print("Digite sair para encerrar.\n")

while True:

    pergunta = input("Pergunta: ")

    if pergunta.lower() == "sair":
        break

    resposta = qa.run(pergunta)

    print("\nResposta:")
    print(resposta)

    print("\n" + "=" * 50)
