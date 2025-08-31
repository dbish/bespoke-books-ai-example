# bespoke-books-ai-example
creating coloring books using AI models :)

for the managed/consumer version of this goal, go here -> https://bespokebooks.io

### Quickstart/Prep

1) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Upgrade pip and install requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Make the venv available as a Jupyter kernel (so the notebook uses it):

```bash
python -m ipykernel install --user --name bespoke-books-venv --display-name "Python (bespoke-books)"
```

### Run the notebook with this venv

- Open `core_coloring_book.ipynb` in your editor (e.g., VS Code) or Jupyter.
- When prompted for a kernel, select "Python (bespoke-books)" (or the kernel you named above).
- If already open, you can switch kernels via the notebook's kernel selector.

To deactivate the venv later:

```bash
deactivate
```
