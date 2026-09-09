# Continuous integration

`ci.yml` runs on pull requests, pushes to `main` / `master`, and manual dispatch.
It does not publish packages or require cloud credentials.

- Python 3.10, 3.11 and 3.12 run the SDK and collaboration tests with LangChain
  and LangGraph installed (the event tests use their native message classes).
- Separate Python 3.11 jobs also install AgentScope, Google ADK, Pydantic AI or
  CrewAI. Tests for other optional frameworks are skipped in each job.
- Tests cover unit behavior, local HTTP/MCP services, framework execution and
  AG-UI / OpenAI protocol output. These are not cloud end-to-end tests.
- Both packages' wheels and source distributions are built and checked by Twine.

To reproduce the common test job:

```bash
python -m pip install -e '.[dev,mcp,server,credentials,langchain,langgraph]' -e ./packages/collaboration
python -m pip check
python -m pytest tests/unit tests/integration tests/optional -q --tb=short
```

Add the relevant framework extra to the install command to reproduce its job.
The tag-triggered PyPI workflow remains separate; see [RELEASING.md](RELEASING.md).
