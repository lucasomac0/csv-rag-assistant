"""
rag_csv.py
----------
RAG (Retrieval-Augmented Generation) sobre um arquivo CSV usando LangChain.

Lê um CSV (ex: catálogo de produtos / avaliações de clientes), transforma cada
linha em um documento, gera embeddings, indexa em um vector store (Chroma) e
usa um LLM para responder perguntas em linguagem natural sobre os dados.

Suporta dois provedores de LLM, escolhidos via --provider ou variável de
ambiente RAG_PROVIDER:

  * gemini  -> Google Gemini (API gratuita via Google AI Studio)
  * ollama  -> modelo local via Ollama (ex: Llama 3), sem custo e sem
               enviar seus dados para a internet

Uso rápido:
    python rag_csv.py --provider gemini
    python rag_csv.py --provider ollama
    python rag_csv.py --provider gemini --question "Qual produto tem mais reclamações?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

DEFAULT_CSV_PATH = "data/avaliacoes_produtos.csv"
DEFAULT_PERSIST_DIR = "./chroma_db"

RAG_PROMPT_TEMPLATE = """Você é um assistente que responde perguntas SOMENTE
com base no contexto abaixo, extraído de um arquivo CSV de produtos/avaliações.

Regras:
- Use apenas as informações do contexto para responder.
- Se a resposta não estiver no contexto, diga claramente que não encontrou
  essa informação nos dados, em vez de inventar.
- Responda em português, de forma direta e objetiva.
- Quando fizer sentido, cite o nome do produto envolvido.

Contexto:
{context}

Pergunta: {question}

Resposta:"""


# Carregamento e preparação dos dados

def load_documents_from_csv(csv_path: str) -> list[Document]:
    """Lê o CSV com pandas e converte cada linha em um Document do LangChain.

    Cada linha vira um texto legível (ex: "Produto: X | Categoria: Y | ...")
    para que o embedding capture o significado de todos os campos daquela
    linha, e não apenas de uma coluna isolada.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Arquivo CSV não encontrado: {csv_path}\n"
            "Passe o caminho correto com --csv caminho/do/arquivo.csv"
        )

    df = pd.read_csv(csv_path)
    documents: list[Document] = []

    for row_idx, row in df.iterrows():
        linha_formatada = " | ".join(
            f"{coluna}: {valor}" for coluna, valor in row.items() if pd.notna(valor)
        )
        documents.append(
            Document(
                page_content=linha_formatada,
                metadata={"source": csv_path, "row": int(row_idx)},
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Quebra documentos muito longos (ex: comentários extensos) em pedaços
    menores. Para linhas curtas de CSV isso normalmente não altera nada,
    mas protege o pipeline caso o CSV tenha colunas de texto longo."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n", " | ", ". ", " "],
    )
    return splitter.split_documents(documents)


# Configuração do provedor de LLM/embeddings (Gemini ou Ollama)

def get_embeddings(provider: str) -> Embeddings:
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY não definida. Crie uma chave gratuita em "
                "https://aistudio.google.com/apikey e coloque-a no arquivo .env"
            )
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=model, base_url=base_url)

    raise ValueError(f"Provedor desconhecido: {provider}")


def get_llm(provider: str) -> BaseChatModel:
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY não definida. Crie uma chave gratuita em "
                "https://aistudio.google.com/apikey e coloque-a no arquivo .env"
            )
        # "gemini-flash-latest" aponta sempre para o modelo Flash mais atual
        # do momento (gratuito no free tier do Google AI Studio). Se preferir
        # travar em uma versão específica, defina GEMINI_CHAT_MODEL no .env.
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest")
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0.2)

    raise ValueError(f"Provedor desconhecido: {provider}")


# Vector store / índice

def build_or_load_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_dir: str,
    provider: str,
    rebuild: bool,
) -> Chroma:
    """Cria (ou reaproveita) o índice vetorial no disco.

    Usa uma coleção por provedor porque Gemini e Ollama geram embeddings com
    dimensões/espaços diferentes — misturá-los na mesma coleção quebraria a
    busca por similaridade.
    """
    collection_name = f"rag_csv_{provider}"
    already_exists = Path(persist_dir).exists() and any(Path(persist_dir).iterdir())

    if already_exists and not rebuild:
        print(f"Reutilizando índice existente em '{persist_dir}' (use --rebuild para recriar).")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    print(f"Gerando embeddings e criando índice vetorial em '{persist_dir}'...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


# Cadeia RAG (LCEL)

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG sobre um arquivo CSV usando LangChain + Gemini/Ollama."
    )
    parser.add_argument(
        "--csv",
        default=os.getenv("RAG_CSV_PATH", DEFAULT_CSV_PATH),
        help=f"Caminho do arquivo CSV (padrão: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default=os.getenv("RAG_PROVIDER", "gemini"),
        help="Provedor de LLM/embeddings a usar (padrão: gemini)",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("RAG_PERSIST_DIR", DEFAULT_PERSIST_DIR),
        help=f"Pasta onde o índice vetorial é salvo (padrão: {DEFAULT_PERSIST_DIR})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Número de trechos do CSV recuperados por pergunta (padrão: 4)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Força a recriação do índice vetorial a partir do CSV.",
    )
    parser.add_argument(
        "--question",
        help="Faz uma única pergunta e sai (modo não interativo). "
        "Se omitido, entra em modo de chat interativo.",
    )
    return parser.parse_args()


def run_single_question(chain, question: str) -> None:
    resposta = chain.invoke(question)
    print(f"\nPergunta: {question}\nResposta: {resposta}\n")


def run_interactive_loop(chain) -> None:
    print("\nDigite sua pergunta sobre os dados do CSV (ou 'sair' para encerrar).\n")
    while True:
        try:
            question = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break

        try:
            resposta = chain.invoke(question)
        except Exception as exc:  # mantém o loop vivo mesmo se uma chamada falhar
            print(f"[Erro ao gerar resposta: {exc}]\n")
            continue

        print(f"Assistente: {resposta}\n")


def main() -> None:
    load_dotenv()
    args = parse_args()

    print(f"Provedor selecionado: {args.provider}")
    print(f"Arquivo CSV: {args.csv}")

    try:
        documents = load_documents_from_csv(args.csv)
        chunks = split_documents(documents)
        print(f"{len(documents)} linha(s) do CSV carregada(s) -> {len(chunks)} trecho(s) indexado(s).")

        embeddings = get_embeddings(args.provider)
        vectorstore = build_or_load_vectorstore(
            chunks, embeddings, args.persist_dir, args.provider, args.rebuild
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": args.top_k})

        llm = get_llm(args.provider)
        chain = build_rag_chain(retriever, llm)
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        print(f"\nErro de configuração: {exc}\n", file=sys.stderr)
        sys.exit(1)

    if args.question:
        run_single_question(chain, args.question)
    else:
        run_interactive_loop(chain)


if __name__ == "__main__":
    main()
