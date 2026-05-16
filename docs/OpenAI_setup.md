## Team setup note: Ollama / OpenAI model selection

This branch adds runtime LLM provider selection to the MCP client.

When running:

```bash
uv run src/agenticcybersense/mcp/main.py
```

the app asks which model provider to use:

```text
1) Local Ollama - llama3.1:8b
2) OpenAI - gpt-5.4-mini
```

For local usage, select `1`.
For OpenAI API usage, select `2`.

### OpenAI API setup

OpenAI API billing is separate from ChatGPT Plus/Pro/Business subscriptions, so each developer needs access to an OpenAI Platform API key and API billing/quota. ([OpenAI Help Center][1])

Steps:

1. Go to the OpenAI Platform dashboard.
2. Open **API keys**.
3. Click **Create new secret key**.
4. Use:

   * **Owner:** You
   * **Project:** Default project, or project assigned by the team
   * **Permissions:** All for local testing
5. Copy the generated key once. Do not commit it to GitHub.

Add this to your local `.env` file:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5.4-mini
```

Optional default provider:

```env
LLM_PROVIDER=ollama
```

or, to default to OpenAI:

```env
LLM_PROVIDER=openai
```

Make sure `.env` is not committed. It should be ignored by Git.

### Cost estimate

We are using:

```env
OPENAI_MODEL=gpt-5.4-mini
```

Current standard pricing for `gpt-5.4-mini` is **$0.75 / 1M input tokens**, **$0.075 / 1M cached input tokens**, and **$4.50 / 1M output tokens**. ([OpenAI Platform][2])

A sample MCP run using `rag_search` + `telegram_search` consumed approximately:

```text
Input tokens: 3703
Cached input tokens: 1152
Output tokens: 252
```

Estimated cost for that run:

```text
≈ $0.0031 USD
```

Without cached-token discount, it would still be around:

```text
≈ $0.0039 USD
```

So a similar query is roughly **$0.003–$0.004 per run**, depending on prompt size, tool output size, and caching.

For rough planning:

```text
100 similar runs ≈ $0.31–$0.39
1000 similar runs ≈ $3.10–$3.90
```

Note: current MCP tools run locally. If we later use OpenAI-hosted tools like web search, those may add separate tool charges. OpenAI’s pricing page lists web search pricing separately. ([OpenAI Platform][2])

---
