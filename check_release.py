from pathlib import Path
import csv
import py_compile

ROOT = Path(__file__).resolve().parent

required = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "ifclass.py",
    "test_ifclass.py",
]

missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")

for name in ["ifclass.py", "test_ifclass.py", "audit_labels.py"]:
    p = ROOT / name
    if p.exists():
        py_compile.compile(str(p), doraise=True)

print("Release structure check passed.")
