# 🛡️ AgenticCyberSense

**Agentic Cyber Threat Intelligence Platform**

AgenticCyberSense is an AI-powered cyber threat intelligence platform that uses multiple specialized agents to monitor, analyze, and report on security threats from various sources including documentation, websites, and Telegram channels.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Agent System](#agent-system)
- [Graph State Flow](#graph-state-flow)
- [Web Crawler & RAG Pipeline](#web-crawler--rag-pipeline)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)

---

## 🏗️ Architecture General Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OpenWebUI (User Interface)                      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ HTTP POST /v1/chat/completions
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Server (FastAPI)                            │
│                              Port: 7001                                      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Orchestration                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        GraphState                                    │    │
│  │  • query              • agents_consulted    • findings              │    │
│  │  • conversation_id    • pending_agents      • final_response        │    │
│  │  • context            • agent_responses     • is_complete           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Documentation │       │     Web       │       │   Telegram    │
│    Agent      │       │    Agent      │       │    Agent      │
│   (RAG)       │       │   (OSINT)     │       │   (OSINT)     │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  ChromaDB     │       │  Web Sources  │       │   Telegram    │
│  (Vector DB)  │       │  (News, CVE)  │       │   Channels    │
└───────────────┘       └───────────────┘       └───────────────┘
```

---

## 🤖 Agent System

### Agent Hierarchy

```
                    ┌─────────────────────┐
                    │    Orchestrator     │
                    │       Agent         │
                    │  (Coordinator)      │
                    └──────────┬──────────┘
                               │
                               │ Always consults first
                               ▼
                    ┌─────────────────────┐
                    │   Documentation     │
                    │       Agent         │
                    │   (RAG-based)       │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Web Agent  │     │  Telegram   │     │   Future    │
    │             │     │   Agent     │     │   Agents    │
    └─────────────┘     └─────────────┘     └─────────────┘
```

### Agent Descriptions

| Agent | Purpose | Data Sources |
|-------|---------|--------------|
| **Orchestrator** | Coordinates all agents, routes queries, synthesizes results | All agent responses |
| **Documentation** | RAG-based knowledge retrieval for security topics | PDF documents, ChromaDB |
| **Web** | Monitors websites for security news and threat intelligence | NIST NVD, CISA, Security blogs |
| **Telegram** | Monitors Telegram channels for threat actor activity | Telegram groups/channels |

### Agent Details

#### 🎯 Orchestrator Agent
- **Role**: Main coordinator
- **Always**: Consults Documentation Agent first
- **Then**: Routes to specialized agents based on query keywords
- **Finally**: Synthesizes all findings into a comprehensive report

#### 📚 Documentation Agent
- **Role**: Knowledge base retrieval
- **Technology**: RAG (Retrieval Augmented Generation)
- **Storage**: ChromaDB vector database
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Sources**: PDF documents, security guides, CVE databases

#### 🌐 Web Agent
- **Role**: Web-based threat intelligence via crawler RAG index
- **Technology**: Semantic search over pre-crawled content
- **Storage**: ChromaDB `webcrawler_pages` collection
- **Sources**: 36 security sites including NIST NVD, CISA, Krebs on Security, BleepingComputer, SOCRadar, GitHub CTI repositories
- **Updated**: Automatically after each crawler run (daily at 02:00)

#### 📱 Telegram Agent
- **Role**: Social media threat intelligence
- **Sources**: Configured Telegram channels/groups
- **Capabilities**:
  - Monitor for leaked credentials
  - Track threat actor communications
  - Detect data breach announcements

---

## 🔄 Graph State Flow

The system uses LangGraph for orchestration. Here's how the state flows through the system:

### State Machine Diagram

```
                                    START
                                      │
                                      ▼
                            ┌─────────────────┐
                            │   Orchestrator  │
                            │      Node       │
                            │                 │
                            │ • Parse query   │
                            │ • Determine     │
                            │   agents needed │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  Documentation  │
                            │      Node       │
                            │                 │
                            │ • Retrieve docs │
                            │ • Build context │
                            └────────┬────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │    Web Node     │    │  Telegram Node  │    │  (Future Nodes) │
    │                 │    │                 │    │                 │
    │ • Query ChromaDB│    │ • Check channels│    │                 │
    │ • Semantic search    │ • Analyze msgs  │    │                 │
    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │   Synthesize    │
                          │      Node       │
                          │                 │
                          │ • Combine all   │
                          │   responses     │
                          │ • Format report │
                          │ • Rank findings │
                          └────────┬────────┘
                                   │
                                   ▼
                                  END
```

### GraphState Object

```python
@dataclass
class GraphState:
    # INPUT
    query: str                    # User's question
    conversation_id: str          # Unique conversation identifier
    context: dict                 # Additional context from history

    # PROCESSING STATE
    current_agent: str            # Currently active agent
    agents_consulted: list[str]   # Agents that have responded
    pending_agents: list[str]     # Agents still to be consulted

    # AGENT RESPONSES
    agent_responses: dict         # {agent_name: AgentResponse}
    documentation_context: str    # Always stored separately

    # RESULTS
    findings: list[Finding]       # All findings from all agents
    final_response: str           # Synthesized final report
    is_complete: bool             # Processing complete flag
    error: str | None             # Error message if any
```

---

## 🕷️ Web Crawler & RAG Pipeline

AgenticCyberSense includes a built-in web crawler that periodically collects threat intelligence from monitored security sites and indexes the content into ChromaDB for semantic search.

### How It Works

```
Scheduler (daily 02:00)
        ↓
main_trafilatura.py          — crawls configured security sites
        ↓
web_crawler/output/
latest_results.json          — raw crawl output (~26MB)
        ↓
rag_ingest.py                — chunks text, embeds, upserts to ChromaDB
        ↓
data/chroma_db/
webcrawler_pages collection  — persistent vector store (16,000+ chunks)
        ↓
Web Agent                    — semantic search on user query
```

### Hash-Based Incremental Crawling

The crawler uses SHA-256 content hashing to avoid re-crawling unchanged pages. On each run, only pages whose content has changed since the last crawl are re-fetched.

```
First run  : ~2-3 hours  (all sites crawled from scratch)
Subsequent : ~20-30 min  (only changed pages re-crawled)
```

Hash history is stored in `web_crawler/output/crawl_history.db` (SQLite).

### Automatic Startup Behavior

When the API server starts:

| Condition | Behavior |
|-----------|----------|
| `latest_results.json` does not exist | Crawler starts immediately |
| `latest_results.json` exists | Crawler waits for next scheduled run (02:00) |

### Running the Crawler Manually

```bash
# Run a full crawl cycle (crawl + automatic RAG ingest)
uv run python src/agenticcybersense/web_crawler/main_trafilatura.py

# Run only the RAG ingest (if JSON already exists)
uv run python src/agenticcybersense/web_crawler/rag_ingest.py

# Check ChromaDB collection sizes
uv run python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/chroma_db')
for col in client.list_collections():
    print(col.name, '->', col.count(), 'chunks')
"
```

### Crawler Configuration

Edit `src/agenticcybersense/web_crawler/config.py`:

```python
CONCURRENT_SITES = 3       # parallel sites (increase for faster crawls)
INACTIVITY_TIMEOUT = 180   # seconds before a site is skipped
FORCE_FULL_CRAWL = False   # True = ignore hashes, re-crawl everything
ENABLE_INCREMENTAL = True  # False = disable hash checks
```

Add or remove URLs in `src/agenticcybersense/web_crawler/config/sites.xlsx`.

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Ollama (for local LLM)
- Playwright Chromium (for JS-heavy sites)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/AgenticCyberSense.git
cd AgenticCyberSense

# Install dependencies
uv sync

# Install Playwright Chromium browser
uv run playwright install chromium

# Copy environment configuration
cp .env.example .env

# Start Ollama (in a separate terminal)
ollama serve
ollama pull llama3.2

# Start the API server
uv run uvicorn agenticcybersense.api_server:app --host 0.0.0.0 --port 7001

# Start the MCP server (optional, in a separate terminal)
uv run uvicorn agenticcybersense.mcp.server:app --host 0.0.0.0 --port 8000
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# LLM Settings
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# API Server
API_HOST=0.0.0.0
API_PORT=7001

# RAG Settings
CHROMA_PERSIST_DIR=./data/chroma_db
PDF_DOCS_DIR=./data/documents
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Logging
LOG_LEVEL=INFO
```

---

## 📖 Usage

### With OpenWebUI

1. Start the API server:
   ```bash
        uv run uvicorn agenticcybersense.api_server:app --host 0.0.0.0 --port 7001
   ```

2. Configure OpenWebUI — go to **Admin Panel → Settings → Connections → OpenAI API**:
        - **URL**: `http://host.docker.internal:7001/v1` (if OpenWebUI runs in Docker)
   - **Key**: `sk-any-value`
   - **Model**: `agenticcybersense`

3. Start chatting!

> **Note — Docker Networking**: If OpenWebUI runs in Docker, `localhost` inside the container refers to the container itself, not the host machine. Use `host.docker.internal` instead. On Windows with WSL, also add a port proxy rule:
>
> ```powershell
> # Admin PowerShell
> netsh interface portproxy add v4tov4 `
>   listenport=7001 listenaddress=0.0.0.0 `
>   connectport=7001 connectaddress=<WSL_IP>
> ```
>
> Get your WSL IP: `ip addr show eth0 | grep "inet "`

### With cURL

```bash
curl -X POST http://localhost:7001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-test" \
  -d '{
    "model": "agenticcybersense",
    "messages": [{"role": "user", "content": "What are recent CVE vulnerabilities?"}],
    "stream": false
  }'
```

---

## 📚 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat endpoint (OpenAI compatible) |
| `/v1/agents` | GET | List available agents |

---

## 🧪 Development

```bash
# Install dev dependencies
uv sync --all-groups

# Run tests
uv run pytest -s tests

# Run linting and formatting
uv run pre-commit run --all-files

# Type checking
uv run mypy src --strict
```

---

## 📁 Project Structure

```
AgenticCyberSense/
├── src/agenticcybersense/
│   ├── api_server.py              # FastAPI server + scheduler lifespan
│   ├── settings.py                # Configuration
│   ├── logging_utils.py           # Logging
│   ├── agents/
│   │   ├── base.py                # BaseAgent ABC
│   │   ├── registry.py            # Agent registry
│   │   ├── orchestrator.py        # Orchestrator agent
│   │   ├── documentation.py       # RAG agent (PDF)
│   │   ├── web.py                 # Web agent (crawler RAG)
│   │   └── telegram.py            # Telegram agent
│   ├── graph/
│   │   ├── state.py               # GraphState definition
│   │   ├── build_graph.py         # LangGraph construction
│   │   └── routing.py             # Routing logic
│   ├── web_crawler/
│   │   ├── config/
│   │   │   └── sites.xlsx         # List of URLs to crawl
│   │   ├── output/
│   │   │   ├── latest_results.json  # Crawl output (~26MB)
│   │   │   └── crawl_history.db     # Hash history (SQLite)
│   │   ├── config.py              # Crawler settings
│   │   ├── main_trafilatura.py    # Main crawl entry point
│   │   ├── rag_ingest.py          # JSON → ChromaDB pipeline
│   │   ├── crawler_scheduler.py   # APScheduler wrapper
│   │   ├── crawl_history_manager.py
│   │   ├── deep_crawler_trafilatura.py
│   │   └── trafilatura_ollama_agent.py
│   ├── rag/
│   │   ├── ingest.py              # PDF document ingestion
│   │   └── retriever.py           # Document retrieval
│   ├── schemas/
│   │   ├── messages.py            # Request/Response schemas
│   │   └── findings.py            # Finding/Severity schemas
│   └── llm/
│       ├── factory.py             # LLM creation
│       └── prompts.py             # Prompt templates
├── data/
│   ├── documents/                 # PDF documents for RAG
│   └── chroma_db/
│       ├── pdf_documents          # Documentation agent vector store
│       └── webcrawler_pages       # Web agent vector store (16,000+ chunks)
├── .env
├── pyproject.toml
└── README.md
```

---

## 📄 License

MIT License