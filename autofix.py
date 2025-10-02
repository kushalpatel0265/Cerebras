#!/usr/bin/env python3
import os, re, json, shlex, time, sys, subprocess, pathlib, datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests

# ------------------ ENV & CONSTANTS ------------------
load_dotenv()
REPO_DIR   = pathlib.Path(os.getenv("REPO_DIR", ".")).resolve()
DOCKERFILE = os.getenv("DOCKERFILE", "Dockerfile")
IMAGE_TAG  = os.getenv("IMAGE_TAG", "autofix:trial")
MAX_ITERS  = int(os.getenv("MAX_ITERS", "5"))
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-4-scout-17b-16e-instruct")

RUNS_DIR = REPO_DIR / "autofix_runs"
RUNS_DIR.mkdir(exist_ok=True, parents=True)

# ------------------ CEREBRAS CLIENT ------------------
cerebras_client = None
try:
    from cerebras.cloud.sdk import Cerebras
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if api_key:
        cerebras_client = Cerebras(api_key=api_key)
except Exception as e:
    print(f"[warn] Cerebras SDK unavailable or API key missing: {e}", file=sys.stderr)

# ------------------ DATA MODELS ------------------
class BuildError(BaseModel):
    file_path: str
    line: Optional[int] = None
    column: Optional[int] = None
    error_code: Optional[str] = None
    severity: str = Field(default="error")
    message: str
    probable_cause: Optional[str] = None

class ErrorExtraction(BaseModel):
    errors: List[BuildError] = Field(default_factory=list)

ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "line": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "column": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "error_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "severity": {"type": "string"},
                    "message": {"type": "string"},
                    "probable_cause": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                },
                "required": ["file_path", "message", "severity"],
                "additionalProperties": False
            }
        }
    },
    "required": ["errors"],
    "additionalProperties": False
}

# ------------------ UTILITIES ------------------
def run(cmd: str, cwd: pathlib.Path = REPO_DIR) -> subprocess.CompletedProcess:
    print(f"\n$ {cmd}")
    return subprocess.run(shlex.split(cmd), cwd=str(cwd), text=True, capture_output=True)

def save_blob(name: str, content: str):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RUNS_DIR / f"{ts}_{name}"
    p.write_text(content, encoding="utf-8")
    print(f"[saved] {p.relative_to(REPO_DIR)}")
    return p

# ------------------ DOCKER BUILD ------------------
def docker_build() -> tuple[bool, str]:
    cmd = f"docker build -f {shlex.quote(DOCKERFILE)} -t {shlex.quote(IMAGE_TAG)} . --progress=plain"
    p = run(cmd, REPO_DIR)
    logs = (p.stdout or "") + "\n" + (p.stderr or "")
    save_blob("build.log", logs)
    return p.returncode == 0, logs

# ------------------ ERROR EXTRACTION ------------------
def naive_extract_errors(log_text: str) -> ErrorExtraction:
    errs: List[BuildError] = []
    pat = re.compile(
        r"(?P<file>[^:\s]+?\.[a-zA-Z0-9_]+):(?P<line>\d+)?(?::(?P<col>\d+))?\s*(?:fatal\s+)?error[:\-\s]+(?P<msg>.+)",
        re.IGNORECASE
    )
    for m in pat.finditer(log_text):
        errs.append(BuildError(
            file_path=m.group("file"),
            line=int(m.group("line")) if m.group("line") else None,
            column=int(m.group("col")) if m.group("col") else None,
            severity="error",
            message=m.group("msg").strip()
        ))
    return ErrorExtraction(errors=errs)

def cerebras_extract_errors(log_text: str) -> ErrorExtraction:
    if not cerebras_client:
        return naive_extract_errors(log_text)

    completion = cerebras_client.chat.completions.create(
        model=CEREBRAS_MODEL,
        messages=[
            {"role": "system", "content": "You are a build-log triage assistant. Extract actionable compile/runtime build errors from Docker build logs. Prefer repository files over system paths; ignore non-actionable noise."},
            {"role": "user", "content": f"From the following Docker build logs, extract errors as STRICT JSON.\n\nLOGS START\n{log_text}\nLOGS END"}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "error_extraction", "strict": True, "schema": ERROR_SCHEMA}
        },
        max_tokens=2048
    )
    content = completion.choices[0].message.content
    save_blob("triage.json", content)
    try:
        data = json.loads(content)
        return ErrorExtraction(**data)
    except Exception as e:
        print(f"[warn] Cerebras returned non-JSON or wrong schema: {e}", file=sys.stderr)
        return naive_extract_errors(log_text)

# ------------------ LLAMA (OLLAMA) ------------------
def get_file_content(path: str) -> str:
    try:
        full = (REPO_DIR / path).resolve()
        if full.exists() and full.is_file():
            return full.read_text(encoding="utf-8")
    except:
        return ""
    return ""

def llama_propose_fixes(errors: ErrorExtraction) -> List[Dict[str, Any]]:
    # prepare context map of files mentioned
    context = {}
    for e in errors.errors:
        context[e.file_path] = get_file_content(e.file_path)

    prompt = f"""
    You are a senior software engineer. You must fix the repository's code to resolve the Docker build errors.

    Your output MUST be valid JSON matching exactly:
    {{
    "patches": [
        {{ "file_path": "string", "new_content": "string" }}
    ]
    }}

    Do not include any text before or after the JSON.
    Return full file content for each patch.

    Errors:
    {errors.model_dump_json(indent=2)}

    Current files:
    {json.dumps(context, indent=2)}
    """


    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False   # we are not using streaming
    }

    try:
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600)
        r.raise_for_status()
    except Exception as e:
        print(f"[err] Ollama request failed: {e}", file=sys.stderr)
        return []

    txt = r.json().get("response", "{}")
    save_blob("llama_raw.json", txt)
    try:
        obj = json.loads(txt)
        return obj.get("patches", [])
    except Exception:
        return []

# ------------------ PATCH APPLICATION ------------------
ALLOWED_ROOTS = {REPO_DIR.resolve()}

def is_safe_path(p: pathlib.Path) -> bool:
    p = p.resolve()
    return any(root == p or root in p.parents for root in ALLOWED_ROOTS)

def apply_patches(patches: List[Dict[str, Any]]) -> int:
    applied = 0
    for patch in patches:
        rel = pathlib.Path(patch["file_path"])
        if rel.is_absolute():           # normalize Windows absolute paths
            rel = pathlib.Path(rel.name)

        if ".." in rel.parts:
            print(f"[skip] Suspicious path: {rel}")
            continue

        dest = (REPO_DIR / rel).resolve()
        if not is_safe_path(dest):
            print(f"[skip] Outside allowed roots: {dest}")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            bak = dest.with_suffix(dest.suffix + ".bak")
            bak.write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
        dest.write_text(patch["new_content"], encoding="utf-8")
        print(f"[ok] Wrote {rel}")
        applied += 1
    return applied

# ------------------ GIT INTEGRATION ------------------
def git_available() -> bool:
    return run("git rev-parse --is-inside-work-tree").returncode == 0

def git_commit(iter_idx: int):
    if not git_available():
        print("[info] Git repo not detected; skipping commit.")
        return
    run("git checkout -B autofix", REPO_DIR)
    run("git add -A", REPO_DIR)
    run(f'git commit -m "autofix iteration {iter_idx}"', REPO_DIR)

# ------------------ MAIN LOOP ------------------
def main() -> int:
    for i in range(1, MAX_ITERS + 1):
        print(f"\n=== Iteration {i}/{MAX_ITERS} ===")
        ok, logs = docker_build()
        if ok:
            print("✅ Build succeeded!")
            return 0

        print("❌ Build failed. Extracting errors...")
        extraction = cerebras_extract_errors(logs)
        if not extraction.errors:
            print("No actionable repo errors found. Stopping.")
            return 2

        print(f"Found {len(extraction.errors)} error(s). Asking Llama for fixes...")
        patches = llama_propose_fixes(extraction)
        if not patches:
            print("Model returned no patches. Stopping.")
            return 3

        n = apply_patches(patches)
        save_blob("applied_patches_count.txt", str(n))
        if n == 0:
            print("No patches applied. Stopping.")
            return 4

        git_commit(i)
        time.sleep(1)

    print("Reached max iterations without success.")
    return 5

if __name__ == "__main__":
    sys.exit(main())
