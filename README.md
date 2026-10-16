# RAG sobre CSV com LangChain (Gemini ou Ollama/Llama 3)

Projeto simples de **RAG (Retrieval-Augmented Generation)** que lê um arquivo
CSV — neste exemplo, avaliações de clientes sobre um catálogo de produtos
fictício — e permite fazer perguntas em linguagem natural sobre esses dados.

O pipeline:

1. Lê o CSV com **pandas** e transforma cada linha em um documento de texto.
2. Gera **embeddings** desses documentos.
3. Indexa os embeddings em um vector store local (**Chroma**).
4. Na hora da pergunta, busca os trechos mais relevantes do CSV (retrieval)
   e envia esse contexto + a pergunta para um **LLM**, que responde com base
   apenas nos dados recuperados (generation).

Funciona com dois provedores de LLM, à sua escolha:

| Provedor | Custo | Onde roda | Requisito |
|---|---|---|---|
| `gemini` | Gratuito (free tier) | Nuvem (API do Google) | Chave de API do Google AI Studio |
| `ollama` | Gratuito | 100% local | [Ollama](https://ollama.com) instalado + modelo baixado |

## Estrutura do projeto

```
llm-rag-csv/
├── data/
│   └── avaliacoes_produtos.csv   # dados de exemplo (troque pelo seu CSV)
├── rag_csv.py                    # script principal
├── requirements.txt
├── .env.example
└── README.md
```

## Pré-requisitos

- Python 3.10+
- Uma das opções abaixo:
  - Uma chave de API gratuita do **Google Gemini**, **ou**
  - **Ollama** instalado localmente

## Instalação

```bash
git clone <url-deste-repositorio>
cd llm-rag-csv

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copie o arquivo de exemplo de variáveis de ambiente:

```bash
cp .env.example .env
```

### Opção A: usando o Google Gemini (gratuito, via nuvem)

1. Gere uma chave gratuita em [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Cole a chave no arquivo `.env`:
   ```
   GOOGLE_API_KEY=sua-chave-aqui
   ```
3. Rode o script (o Gemini já é o provedor padrão):
   ```bash
   python rag_csv.py
   ```

> O script usa por padrão `gemini-flash-latest`, que sempre aponta para o Flash mais recente
> disponível no free tier. Se quiser escolher uma versão específica, defina
> `GEMINI_CHAT_MODEL` no `.env`.

### Opção B: usando Ollama (local)

1. Instale o [Ollama](https://ollama.com/download).
2. Verifique as configurações da sua máquina e escolha um modelo adequado (ex: Llama 3):
   ```bash
   ollama pull llama3
   ```
3. Garanta que o Ollama está rodando (`ollama serve`, ou já roda em segundo
   plano após a instalação).
4. Rode o script apontando para o provedor `ollama`:
   ```bash
   python rag_csv.py --provider ollama
   ```

## Uso

Modo interativo (chat no terminal):

```bash
python rag_csv.py --provider gemini
```

```
Você: Qual produto tem a melhor avaliação média?
Assistente: ...

Você: Existe alguma reclamação sobre bateria?
Assistente: ...

Você: sair
```

Modo de pergunta única (não interativo, útil para scripts/testes):

```bash
python rag_csv.py --provider gemini --question "Quais produtos tiveram nota 1 ou 2?"
```

Usando seu próprio CSV:

```bash
python rag_csv.py --provider gemini --csv data/meu_catalogo.csv --rebuild
```

### Principais argumentos

| Argumento | Descrição | Padrão |
|---|---|---|
| `--provider` | `gemini` ou `ollama` | `gemini` |
| `--csv` | Caminho do arquivo CSV | `data/avaliacoes_produtos.csv` |
| `--top-k` | Quantos trechos do CSV recuperar por pergunta | `4` |
| `--rebuild` | Recria o índice vetorial do zero (use ao trocar de CSV) | desligado |
| `--question` | Faz uma pergunta e sai, sem abrir o chat interativo | — |
| `--persist-dir` | Pasta onde o índice vetorial fica salvo | `./chroma_db` |

## Usando seus próprios dados

Basta trocar o arquivo em `data/` (ou apontar `--csv` para outro caminho) —
qualquer CSV com cabeçalho funciona, seja um catálogo de produtos, uma base
de tickets de suporte, avaliações de clientes, etc. Cada linha do CSV vira um
"documento" pesquisável. Depois de trocar o CSV, rode com `--rebuild` para
recriar o índice vetorial.

## Como o RAG funciona neste projeto (resumo)

1. **Carregar**: cada linha do CSV vira um texto tipo
   `produto: X | categoria: Y | comentario: ...`.
2. **Indexar**: esse texto é convertido em vetor (embedding) e guardado no
   Chroma, um banco vetorial local que persiste em disco.
3. **Buscar (Retrieval)**: quando você faz uma pergunta, o sistema busca os
   `top-k` trechos do CSV mais parecidos semanticamente com a pergunta.
4. **Gerar (Generation)**: esses trechos + a pergunta são enviados ao LLM
   (Gemini ou Llama 3 via Ollama), que responde com base apenas nesse
   contexto — reduzindo alucinações em relação a perguntar direto ao modelo
   sem contexto.

## Licença

Sinta-se livre para usar, modificar e adaptar este projeto (ex: licença MIT).
