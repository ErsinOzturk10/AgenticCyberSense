
# GitHub Workflow Checklist (VS Code + uv)

> Goal: Clone the repo → set up the environment → work on a **feature branch** → run quality checks → commit/push → PR → merge → clean up branches.

---

## ✅ 1) Clone the repo and open the correct folder

- [ ] Create a new working folder on your computer.
- [ ] Open VS Code in this folder.
- [ ] Clone the repo in the terminal:

```bash
git clone https://github.com/ErsinOzturk10/AgenticCyberSense.git
````

* [ ] Get the correct clone link from GitHub:

  * [ ] Go to the repo page and open the **Code** tab.
  * [ ] Make sure the selected branch is **main** (if not, select **main**).
  * [ ] Copy the **Code → HTTPS** link.

* [ ] After a successful clone, repo files will appear on the left side of VS Code.

* [ ] **Important:** If `git branch` shows nothing, you are probably in the **wrong directory**.

  * [ ] In VS Code, open the cloned project folder via **File → Open Folder**:

    * Example: `.../your_work_folder/AgenticCyberSense`
  * [ ] Verify the folder name in the top-left matches the project name.

---

## ✅ 2) Set up the environment (uv + venv)

* [ ] Sync dependencies:

```bash
uv sync
```

* [ ] Verify the `.venv/` folder is created.
* [ ] Activate the virtual environment:

**Windows**

```bash
.\.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

* [ ] Confirm you see the venv name in parentheses `(… )` at the start of the terminal prompt.
* [ ] Verify the active branch:

```bash
git branch
```

Expected: `* main`

---

## ✅ 3) Create a feature branch (do not work on main)

* [ ] **Do not develop on `main`.** Create a **new branch from `main`** for each task.
* [ ] Example: create a branch for the RAG task:

```bash
git checkout -b feat/rag
```

* [ ] Check:

```bash
git branch
```

Expected:

* `* feat/rag`
* `  main`

---

## ✅ 4) Track code changes (VS Code)

* [ ] Open **Source Control** in VS Code:

  * Shortcut: `Ctrl + Shift + G`
* [ ] See changed files under **Changes**.
* [ ] Click a file to view the diff:

  * Left: `main`
  * Right: `feat/...`
    showing the differences.

---

## ✅ 5) Stage changes (before push)

* [ ] Stage all changes:

```bash
git add .
```

* [ ] Stage a single file:

```bash
git add filename.ext
```

* [ ] Alternative (VS Code):

  * [ ] Click `+` next to **Changes** → stage all
  * [ ] Click `+` next to a file → stage only that file

---

## ✅ 6) Quality checks (before push)

> Note: These steps help you catch locally the same checks that run in GitHub Actions (`build.yml`) after pushing.

* [ ] Run pre-commit checks for staged files:

```bash
uvx pre-commit run --all-files
```

* [ ] **Important:** This command checks **only staged files** (it does not check unstaged changes).
* [ ] Some issues may be auto-fixed. Then run it again:

```bash
uvx pre-commit run --all-files
```

* [ ] Do not continue until you see `Passed` for `ruff check` and `ruff format`.
* [ ] If you must ignore a warning in an exceptional case, add this to the end of the line:

```python
# noqa: T201
```

> Replace `T201` with the warning code shown in the terminal. (Use only in exceptional cases agreed within the team.)

* [ ] Type checking (mypy):

```bash
uv run mypy src --strict
```

Expected (if no issues): `Success: no issues found ...`

---

## ✅ 7) Commit + Push (Publish / Sync)

* [ ] Write a short, clear commit message.
* [ ] Commit:

```bash
git commit -m "feat: rag pipeline skeleton"
```

> You can also commit from the VS Code UI (Source Control message box + Commit button).

* [ ] If this is the first push for the branch:

```bash
git push -u origin feat/rag
```

> In VS Code this usually appears as **Publish Branch**.
> If you’ve pushed before, you’ll typically see **Sync**.

---

## ✅ 8) Open a Pull Request on GitHub

* [ ] Go to the repo → **Pull requests** tab.

* [ ] Use the `Compare & pull request` button.

* [ ] Check if **Build** is running in Actions:

  * Repo → **Actions** → workflow runs

* [ ] Create the PR after the build completes successfully.

* [ ] Add **Reviewers** from the top-right.

* [ ] Click `Create a pull request`.

* [ ] If a reviewer requests changes:

  * [ ] Fix locally → stage → run checks → commit → push
  * [ ] The PR updates automatically.

* [ ] After at least 1 approval:

  * [ ] Merge into `main` via `Merge pull request`
  * [ ] After merge, delete the feature branch via `Delete branch`

---

## ✅ 9) After merge: update local main and start the next task

* [ ] Switch back to `main` locally:

```bash
git checkout main
```

* [ ] Pull the latest `main` from remote:

```bash
git pull origin main
```

* [ ] Create a new feature branch for the next task:

```bash
git checkout -b feat/mcp
```

* [ ] Verify:

```bash
git branch
```

Expected: `* feat/mcp`

---

## ✅ 10) Merge conflict note

* [ ] If different changes were made on the same lines, you may get a **merge conflict**.
* [ ] Resolution: the people involved decide together:

  * [ ] Which change should remain?
  * [ ] Should both be kept?
  * [ ] Validate with tests/CI results.

```
