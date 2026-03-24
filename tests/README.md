# Running Tests

This repository uses `pytest`.

## 1. Install Dependencies

From the repository root:

```powershell
pip install -e .[dev]
```

If you prefer a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## 2. Run All Tests

From the repository root:

```powershell
python -m pytest
```

## 3. Run Only Verifier Tests

To run the bait-specific verifier tests:

```powershell
python -m pytest tests\test_bait.py
```

To run the larger Science Bowl verifier question suite:

```powershell
python -m pytest tests\test_verifier_questions.py
```

To run both verifier-related suites together:

```powershell
python -m pytest tests\test_bait.py tests\test_verifier_questions.py
```

## 4. Useful Pytest Options

Show verbose test names:

```powershell
python -m pytest -v
```

Stop on first failure:

```powershell
python -m pytest -x
```

Run only tests whose names match `bait`:

```powershell
python -m pytest -k bait
```

## Notes

- Run commands from the repository root: `C:\Users\gidit\scibowl-gpt`
- The package is configured in editable mode through [pyproject.toml](C:/Users/gidit/scibowl-gpt/pyproject.toml)
- If imports fail, make sure the editable install completed successfully and that your virtual environment is activated
