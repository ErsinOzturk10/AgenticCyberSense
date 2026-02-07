# 🛡️ AgenticCyberSense

**Agentic Cyber Threat Intelligence Platform**

AgenticCyberSense is an AI-powered cyber threat intelligence platform that uses multiple specialized agents to monitor, analyze, and report on security threats from various sources including documentation, websites, and Telegram channels.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Agent System](#agent-system)
- [Graph State Flow](#graph-state-flow)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)

---

## 🏗️ Architecture Overview

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
- **Role**: Web-based threat intelligence
- **Sources**:
  - NIST National Vulnerability Database (NVD)
  - CISA Security Alerts
  - Security news sites (Krebs, Hacker News, BleepingComputer)
- **Capabilities**: CVE lookup, breach monitoring, security news

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
                            │  Documentation  │◄─────────────────┐
                            │      Node       │                  │
                            │                 │                  │
                            │ • Retrieve docs │      ALWAYS      │
                            │ • Build context │      FIRST       │
                            └────────┬────────┘                  │
                                     │                           │
                         ┌───────────┴───────────┐               │
                         │       Router          │───────────────┘
                         │                       │
                         │  if "documentation"   │
                         │  not in consulted     │
                         └───────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │    Web Node     │    │  Telegram Node  │    │  (Future Nodes) │
    │                 │    │                 │    │                 │
    │ • Check sources │    │ • Check channels│    │                 │
    │ • Find threats  │    │ • Analyze msgs  │    │                 │
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

The `GraphState` dataclass holds all information as it flows through the graph:

```python
@dataclass
class GraphState:
    # ═══════════════════════════════════════════════════════════
    # INPUT
    # ═══════════════════════════════════════════════════════════
    query: str                    # User's question
    conversation_id: str          # Unique conversation identifier
    context: dict                 # Additional context from history
    
    # ═══════════════════════════════════════════════════════════
    # PROCESSING STATE
    current_agent: str            # Currently active agent
    agents_consulted: list[str]   # Agents that have responded
    pending_agents: list[str]     # Agents still to be consulted
    
    # ═══════════════════════════════════════════════════════════
    # AGENT RESPONSES
    agent_responses: dict         # {agent_name: AgentResponse}
    documentation_context: str    # Special: always stored separately
    
    # ═══════════════════════════════════════════════════════════
    # RESULTS
    findings: list[Finding]       # All findings from all agents
    final_response: str           # Synthesized final report
    is_complete: bool             # Processing complete flag
    error: str | None             # Error message if any
```

### State Transitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STATE TRANSITIONS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INITIAL STATE                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ query: "What CVE vulnerabilities affect Apache?"                     │   │
│  │ agents_consulted: []                                                 │   │
│  │ pending_agents: []                                                   │   │
│  │ findings: []                                                         │   │
│  │ is_complete: false                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  AFTER ORCHESTRATOR                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ query: "What CVE vulnerabilities affect Apache?"                     │   │
│  │ agents_consulted: []                                                 │   │
│  │ pending_agents: ["web", "telegram"]  ◄── Determined from keywords   │   │
│  │ findings: []                                                         │   │
│  │ is_complete: false                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  AFTER DOCUMENTATION                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ query: "What CVE vulnerabilities affect Apache?"                     │   │
│  │ agents_consulted: ["documentation"]  ◄── Added                       │   │
│  │ pending_agents: ["web", "telegram"]                                  │   │
│  │ documentation_context: "CVE info about Apache..."  ◄── Stored        │   │
│  │ findings: [Finding(title="Doc Reference...")]  ◄── Added             │   │
│  │ is_complete: false                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  AFTER WEB AGENT                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ agents_consulted: ["documentation", "web"]  ◄── Added                │   │
│  │ pending_agents: ["telegram"]  ◄── Removed "web"                      │   │
│  │ findings: [Finding(...), Finding("Web Intel: NIST")]  ◄── Added      │   │
│  │ is_complete: false                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  AFTER TELEGRAM AGENT                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ agents_consulted: ["documentation", "web", "telegram"]               │   │
│  │ pending_agents: []  ◄── Empty, all agents consulted                  │   │
│  │ findings: [Finding(...), Finding(...), Finding("Telegram...")]       │   │
│  │ is_complete: false                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  FINAL STATE (After Synthesize)                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ agents_consulted: ["documentation", "web", "telegram"]               │   │
│  │ pending_agents: []                                                   │   │
│  │ findings: [Finding(...), Finding(...), Finding(...)]                 │   │
│  │ final_response: "# 🛡️ Cyber Threat Intelligence Report..."          │   │
│  │ is_complete: true  ◄── Done!                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Router Logic

```
┌─────────────────────────────────────────────────────────────────┐
│                      ROUTER DECISION TREE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                         Router Called                            │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  Has error?     │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│              ┌──────────────┴──────────────┐                    │
│              │ YES                         │ NO                  │
│              ▼                             ▼                     │
│     Return "synthesize"         ┌─────────────────┐             │
│                                 │ "documentation"  │             │
│                                 │ in consulted?    │             │
│                                 └────────┬────────┘             │
│                                          │                       │
│                           ┌──────────────┴──────────────┐       │
│                           │ NO                          │ YES    │
│                           ▼                             ▼        │
│                  Return "documentation"      ┌─────────────────┐│
│                                              │ pending_agents  ││
│                                              │ not empty?      ││
│                                              └────────┬────────┘│
│                                                       │         │
│                                        ┌──────────────┴─────┐   │
│                                        │ YES               │NO  │
│                                        ▼                   ▼    │
│                               Return next agent    Return       │
│                               from pending         "synthesize" │
│                               ("web"/"telegram")                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Ollama (for local LLM)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/AgenticCyberSense.git
cd AgenticCyberSense

# Install dependencies
uv sync

# Copy environment configuration
cp .env.example .env

# Start Ollama (in another terminal)
ollama serve
ollama pull llama3.2

# Run the server
uv run python -m agenticcybersense.api_server
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
   uv run python -m agenticcybersense.api_server
   ```

2. Configure OpenWebUI:
   - **Base URL**: `http://localhost:7001/v1`
   - **API Key**: `sk-dummy` (any value)
   - **Model**: `agenticcybersense`

3. Start chatting!

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

# Run linting
uvx pre-commit run --all-files

# Type checking
uv run mypy src --strict
```

---

## 📁 Project Structure

```
AgenticCyberSense/
├── src/agenticcybersense/
│   ├── __init__.py
│   ├── api_server.py          # FastAPI server
│   ├── settings.py            # Configuration
│   ├── logging_utils.py       # Logging
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseAgent ABC
│   │   ├── registry.py        # Agent registry
│   │   ├── orchestrator.py    # Orchestrator agent
│   │   ├── documentation.py   # RAG agent
│   │   ├── web.py             # Web intelligence
│   │   └── telegram.py        # Telegram intelligence
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py           # GraphState definition
│   │   ├── build_graph.py     # LangGraph construction
│   │   └── routing.py         # Routing logic
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py          # Document ingestion
│   │   └── retriever.py       # Document retrieval
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── messages.py        # Request/Response schemas
│   │   └── findings.py        # Finding/Severity schemas
│   └── llm/
│       ├── __init__.py
│       ├── factory.py         # LLM creation
│       └── prompts.py         # Prompt templates
├── tests/
├── data/
│   ├── documents/             # PDF documents for RAG
│   └── chroma_db/             # Vector database
├── .env
├── pyproject.toml
└── README.md
```

---

## 📄 License

MIT License