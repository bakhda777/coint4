#!/usr/bin/env python3
import os
import json
import threading
import time
import uuid
import subprocess
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).parent.parent
WEB_DIR = ROOT / "src" / "web_analysis"
LOG_DIR = ROOT / "outputs" / "web_logs"
TASKS = {}
TASKS_FILE = LOG_DIR / "tasks.json"
RUNNING_PROCESSES = {}

def save_tasks():
    try:
        data = {k: v for k, v in TASKS.items()}
        with open(TASKS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving tasks: {e}")

def load_tasks():
    global TASKS
    if TASKS_FILE.exists():
        try:
            with open(TASKS_FILE, 'r') as f:
                TASKS = json.load(f)
        except Exception as e:
            print(f"Error loading tasks: {e}")

load_tasks()

import math

def sanitize_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, obj, code=200):
        # Sanitize object to replace NaN/Inf with null (valid JSON)
        safe_obj = sanitize_json(obj)
        data = json.dumps(safe_obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}
            mode = payload.get("mode", "optimization")
            n_trials = int(payload.get("n_trials", 50))
            
            # Auto-generate study name and path if not provided
            default_name = f"study_{int(time.time())}"
            study_name = payload.get("study_name") or default_name
            
            # Sanitize study name for filename
            safe_name = "".join(c for c in study_name if c.isalnum() or c in ('-', '_')).strip()
            if not safe_name:
                safe_name = default_name
                
            storage_path = f"outputs/studies/{safe_name}.db"
            
            search_space = payload.get("search_space", "configs/search_space_fast.yaml")
            base_config = payload.get("base_config", "configs/main_2024.yaml")
            n_jobs = int(payload.get("n_jobs", -1))
            wf_overrides = payload.get("wf_overrides", {})
            print(f"WF_OVERRIDES_IN_REQUEST: {wf_overrides}")
            wf_overrides_json = json.dumps(wf_overrides).replace("'", "\\'") # Escape single quotes just in case

            LOG_DIR.mkdir(parents=True, exist_ok=True)
            task_id = uuid.uuid4().hex
            log_file = LOG_DIR / f"{task_id}.log"
            TASKS[task_id] = {
                "status": "running",
                "exit_code": None,
                "log_file": str(log_file),
                "mode": mode,
                "wf_overrides": wf_overrides, # Store overrides for history
            }

            def runner():
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "1"
                env["OPENBLAS_NUM_THREADS"] = "1"
                env["NUMEXPR_NUM_THREADS"] = "1"
                cmd = []
                if mode == "optimization":
                    cmd = [
                        env.get("PYTHON", "python"),
                        str(ROOT / "src" / "optimiser" / "run_optimization.py"),
                        "--n-trials", str(n_trials),
                        "--study-name", study_name,
                        "--storage-path", storage_path,
                        "--search-space", search_space,
                        "--base-config", base_config,
                        "--n-jobs", str(n_jobs),
                    ]
                elif mode == "tests":
                    cmd = [
                        "bash", "-lc",
                        "python -m pytest -q --disable-warnings --junitxml=report.xml -v --color=no; rc=$?; echo \"=== PYTEST_DONE exit_code=$rc ===\"; exit 0",
                    ]
                elif mode == "backtest":
                    code = (
                        "import sys, json\n"
                        "from pathlib import Path\n"
                        f"sys.path.insert(0, r'{str(ROOT / 'src')}')\n"
                        "from coint2.utils.config import load_config\n"
                        "from coint2.pipeline.walk_forward_orchestrator import run_walk_forward\n"
                        f"cfg = load_config('{base_config}')\n"
                        f"wf_overrides = json.loads('{wf_overrides_json}')\n"
                        "if wf_overrides.get('start_date'):\n"
                        "    cfg.walk_forward.start_date = wf_overrides['start_date']\n"
                        "if wf_overrides.get('num_steps'):\n"
                        "    cfg.walk_forward.max_steps = int(wf_overrides['num_steps'])\n"
                        "if wf_overrides:\n"
                        "    print(f'WF_OVERRIDES_APPLIED: start_date={cfg.walk_forward.start_date}, num_steps={cfg.walk_forward.max_steps}')\n"
                        "print('=== CONFIGURATION START ===')\n"
                        "try:\n"
                        "    print(cfg.model_dump_json(indent=2))\n"
                        "except AttributeError:\n"
                        "    try:\n"
                        "        print(cfg.json(indent=2))\n"
                        "    except AttributeError:\n"
                        "        print(cfg)\n"
                        "print('=== CONFIGURATION END ===')\n"
                        "res = run_walk_forward(cfg) or {}\n"
                        "print('RESULT_JSON ' + json.dumps(res))\n"
                    )
                    cmd = [env.get("PYTHON", "python"), "-c", code]
                else:
                    cmd = [
                        env.get("PYTHON", "python"),
                        str(ROOT / "scripts" / "optimize_params.py"),
                    ]
                with log_file.open("wb") as lf:
                    lf.write(f"START mode={mode}\n".encode("utf-8"))
                    try:
                        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
                        RUNNING_PROCESSES[task_id] = proc
                        
                        for line in iter(proc.stdout.readline, b""):
                            if not line:
                                break
                            try:
                                s = line.decode("utf-8", "ignore").strip()
                                if s.startswith("RESULT_JSON "):
                                    try:
                                        TASKS[task_id]["result"] = json.loads(s.replace("RESULT_JSON ", "", 1))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            lf.write(line)
                            lf.flush()
                        proc.wait()
                        rc = proc.returncode
                        TASKS[task_id]["exit_code"] = rc
                        TASKS[task_id]["status"] = "done"
                        if task_id in RUNNING_PROCESSES:
                            del RUNNING_PROCESSES[task_id]
                            
                        lf.write(f"\n=== TASK_DONE exit_code={rc} ===\n".encode("utf-8"))
                        save_tasks()
                        lf.flush()
                        if mode == "optimization":
                            try:
                                upd_cmd = [env.get("PYTHON", "python"), str(ROOT / "scripts" / "update_web_analysis.py"), study_name, storage_path, "results"]
                                upd = subprocess.run(upd_cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
                                lf.write(upd.stdout or b"")
                                lf.flush()
                            except Exception as e:
                                lf.write(f"UPDATE_ERROR {e}\n".encode("utf-8"))
                                lf.flush()
                        if mode == "tests":
                            try:
                                import xml.etree.ElementTree as ET
                                rpt = ROOT / "report.xml"
                                if rpt.exists():
                                    tree = ET.parse(str(rpt))
                                    root = tree.getroot()
                                    ts = root.find("testsuite")
                                    if ts is not None:
                                        jr = {
                                            "tests": int(ts.get("tests", "0")),
                                            "failures": int(ts.get("failures", "0")),
                                            "errors": int(ts.get("errors", "0")),
                                            "skipped": int(ts.get("skipped", "0")),
                                            "failed": []
                                        }
                                        for tc in ts.findall("testcase"):
                                            for fl in tc.findall("failure"):
                                                name = tc.get("name", "")
                                                msg = fl.get("message", "")
                                                jr["failed"].append({"name": name, "message": msg})
                                        TASKS[task_id]["junit"] = jr
                            except Exception:
                                pass
                            save_tasks()
                    except Exception as e:
                        TASKS[task_id]["exit_code"] = -1
                        TASKS[task_id]["status"] = "done"
                        if task_id in RUNNING_PROCESSES:
                            del RUNNING_PROCESSES[task_id]
                        lf.write(f"ERROR {e}\n".encode("utf-8"))
                        save_tasks()
                        lf.flush()

            TASKS[task_id]["started_at"] = time.time()
            save_tasks()
            threading.Thread(target=runner, daemon=True).start()
            self._send_json({"task_id": task_id})
            return
        
        if parsed.path == "/api/stop":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}
            
            task_id = payload.get("task_id")
            if not task_id:
                self._send_json({"error": "missing_task_id"}, 400)
                return

            if task_id in RUNNING_PROCESSES:
                proc = RUNNING_PROCESSES[task_id]
                try:
                    import signal
                    # Try terminate first
                    proc.terminate()
                    # Wait a bit, if still running, kill
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    
                    if task_id in TASKS:
                        TASKS[task_id]["status"] = "stopped"
                        TASKS[task_id]["exit_code"] = -1
                        save_tasks()
                    
                    self._send_json({"status": "stopped", "task_id": task_id})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                # Task might be already finished or not found
                if task_id in TASKS and TASKS[task_id]["status"] == "running":
                    # Stale state?
                    TASKS[task_id]["status"] = "done"
                    TASKS[task_id]["exit_code"] = -1
                    save_tasks()
                    self._send_json({"status": "stopped_stale", "task_id": task_id})
                else:
                    self._send_json({"error": "task_not_running_or_found"}, 404)
            return

        if parsed.path == "/api/config/save":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body)
            except Exception:
                self._send_json({"error": "invalid_json"}, 400)
                return
            
            path_arg = payload.get("path")
            content = payload.get("content")
            
            if not path_arg or content is None:
                self._send_json({"error": "missing_fields"}, 400)
                return

            file_path = ROOT / path_arg
            # Security check
            try:
                file_path = file_path.resolve()
                if ROOT not in file_path.parents and file_path != ROOT:
                    self._send_json({"error": "forbidden"}, 403)
                    return
            except Exception:
                 self._send_json({"error": "invalid_path"}, 400)
                 return

            try:
                # Create backup before saving
                if file_path.exists():
                    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                    import shutil
                    shutil.copy2(file_path, backup_path)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                self._send_json({"status": "ok", "path": str(file_path.relative_to(ROOT))})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        return super().do_POST()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            qs = parse_qs(parsed.query)
            task_id = (qs.get("task_id") or [None])[0]
            
            task_info = TASKS.get(task_id)
            if not task_info and TASKS_FILE.exists():
                try:
                    with open(TASKS_FILE, 'r') as f:
                        saved_tasks = json.load(f)
                        task_info = saved_tasks.get(task_id)
                except Exception:
                    pass

            if not task_info:
                self._send_json({"error": "not_found"}, 404)
                return
            self._send_json({"task": task_info})
            return
        if parsed.path == "/api/stream":
            qs = parse_qs(parsed.query)
            task_id = (qs.get("task_id") or [None])[0]
            if not task_id or task_id not in TASKS:
                self.send_response(404)
                self.end_headers()
                return
            log_path = Path(TASKS[task_id]["log_file"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            pos = 0
            while True:
                if log_path.exists():
                    with log_path.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                        if chunk:
                            for line in chunk.splitlines():
                                self.wfile.write(b"data: " + line + b"\n\n")
                                self.wfile.flush()
                            pos = f.tell()
                if TASKS[task_id]["status"] == "done":
                    break
                time.sleep(0.3)
            return
        if parsed.path == "/api/tasks":
            # Prefer loading from disk to ensure history is up to date
            tasks_source = TASKS
            if TASKS_FILE.exists():
                try:
                    with open(TASKS_FILE, 'r') as f:
                        tasks_source = json.load(f)
                except Exception:
                    pass

            items = []
            for tid, info in tasks_source.items():
                it = dict(info)
                it["task_id"] = tid
                items.append(it)
            items.sort(key=lambda x: x.get("started_at", 0), reverse=True)
            self._send_json({"tasks": items})
            return
        if parsed.path == "/api/options":
            configs = []
            search_spaces = []
            studies = []
            
            if (ROOT / "configs").exists():
                for f in (ROOT / "configs").glob("*.yaml"):
                    if "search_space" in f.name:
                        search_spaces.append(str(f.relative_to(ROOT)))
                    else:
                        configs.append(str(f.relative_to(ROOT)))
            
            if (ROOT / "outputs" / "studies").exists():
                for f in (ROOT / "outputs" / "studies").glob("*.db"):
                    studies.append(str(f.relative_to(ROOT)))
            
            self._send_json({
                "configs": sorted(configs),
                "search_spaces": sorted(search_spaces),
                "studies": sorted(studies)
            })
            return
        if parsed.path == "/api/config":
            qs = parse_qs(parsed.query)
            path_arg = (qs.get("path") or [None])[0]
            if not path_arg:
                self._send_json({"error": "missing_path"}, 400)
                return
            
            file_path = ROOT / path_arg
            # Security check: path must be within ROOT and typically inside configs/
            try:
                file_path = file_path.resolve()
                if ROOT not in file_path.parents and file_path != ROOT:
                    self._send_json({"error": "forbidden"}, 403)
                    return
            except Exception:
                 self._send_json({"error": "invalid_path"}, 400)
                 return

            if not file_path.exists() or not file_path.is_file():
                self._send_json({"error": "not_found"}, 404)
                return
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self._send_json({"content": content, "path": str(file_path.relative_to(ROOT))})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return
            
        if parsed.path == "/api/logs/download":
            qs = parse_qs(parsed.query)
            task_id = (qs.get("task_id") or [None])[0]
            
            if not task_id or task_id not in TASKS:
                self.send_error(404, "Log not found")
                return
                
            log_file = Path(TASKS[task_id]["log_file"])
            if not log_file.exists():
                self.send_error(404, "Log file missing")
                return

            try:
                with open(log_file, "rb") as f:
                    content = f.read()
                
                filename = f"task_{task_id}_{TASKS[task_id]['mode']}.log"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, str(e))
            return
            
        return super().do_GET()

def main():
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    print("Server listening on http://localhost:8000/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
