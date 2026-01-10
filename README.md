# AgenticCyberSense
AgenticCyberSense is a project aimed at exploring the intersection of agency and cybersecurity. This repository contains resources, code, and documentation to support the development and understanding of secure systems with a focus on user empowerment and proactive defense mechanisms.

Precommit test before push:

uvx pre-commit run --all-files

uv run mypy src --strict

---

## Installation notes ⚠️

- If you're on **macOS (Intel / x86_64)**, PyTorch wheels for some versions may not be available on PyPI (you may see errors like: "Distribution `torch==2.9.1` can't be installed..."). In that case:
  - Install PyTorch manually via Conda (recommended):

    ```bash
    conda install pytorch -c pytorch
    ```

  - Or follow the official PyTorch install selector at https://pytorch.org/ and choose the command appropriate for your system.

- Project organization:
  - By default the package **does not** install `torch` on macOS Intel to keep `uv sync` working. If you want to opt into the ML extras (install `torch` on platforms where wheels are available), use:

    ```bash
    uv sync .[ml]
    # or with pip
    pip install .[ml]
    ```

  - On **macOS (Intel / x86_64)**, `uv sync .[ml]` will not automatically install `torch` (no compatible wheel on PyPI for some versions). Install PyTorch manually via Conda:

    ```bash
    conda install pytorch -c pytorch
    ```

- If you are on macOS **ARM (Apple Silicon)**, Linux, or Windows, `torch` will be included by default when supported wheels are available.