# Virtual environment (.venv)

This project uses a local virtual environment stored in `.venv`.

Activation (Windows PowerShell):

```powershell
# from the workspace root
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # only if you get a policy error
.\.venv\Scripts\Activate.ps1
```

Activation (Windows CMD):

```cmd
\.venv\Scripts\activate.bat
```

Activation (Git Bash / MSYS):

```bash
source .venv/Scripts/activate
```

Verify the venv Python and pip after activation:

```powershell
python -V
pip --version
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Notes:
- The venv folder is `.venv` at the workspace root.
- Use the venv's python directly without activating when scripting: `.\.venv\Scripts\python.exe script.py`.
