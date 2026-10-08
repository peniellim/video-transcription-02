#!/usr/bin/env python3
"""Colab backend: receives chunked video uploads from the website, runs the
transcription pipeline, and serves the Excel/SRT results. Exposed through a
Cloudflare quick tunnel; every request needs the secret key."""
import contextlib, io, logging, os, re, secrets, shutil, subprocess, threading, time, traceback, urllib.request
from flask import Flask, request, jsonify, send_file, abort
from video_transcriber import process

PORT = 8000
ROOT = "/content/jobs" if os.path.isdir("/content") else os.path.join(os.getcwd(), "jobs")
KEY = secrets.token_urlsafe(12)
JOBS, LOCK = {}, threading.Lock()
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "X-Key, Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r


@app.before_request
def guard():
    if request.method == "OPTIONS":
        return "", 204
    if request.headers.get("X-Key") != KEY:
        return jsonify(error="bad key"), 401


def clean(v):
    return re.sub(r"\W", "", v)


@app.get("/ping")
def ping():
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    return jsonify(ok=True, gpu=gpu)


@app.post("/upload")
def upload():
    d = os.path.join(ROOT, clean(request.args["id"]))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"part{int(request.args['index']):05d}"), "wb") as f:
        f.write(request.get_data())
    return jsonify(ok=True)


class Tee(io.TextIOBase):
    def __init__(self, job):
        self.job, self.buf = job, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.job["log"].append(line.strip())
                del self.job["log"][:-30]
        return len(s)


def run(uid, video, b):
    job = JOBS[uid]
    with LOCK:
        job["state"] = "running"

        def prog(stage, p):
            job["stage"], job["progress"] = stage, round(p, 3)
        try:
            with contextlib.redirect_stdout(Tee(job)):
                x, s, info = process(video, model=b.get("model", "large-v3"), device="auto",
                                     ocr_interval=float(b.get("interval", 1.0)),
                                     do_ocr=bool(b.get("ocr", True)), progress=prog)
            job.update(state="done", progress=1.0, stage="Done", xlsx=x, srt=s, summary=info)
        except Exception as e:
            traceback.print_exc()
            job.update(state="error", stage="Failed", error=str(e))
        finally:
            try:
                os.remove(video)
            except OSError:
                pass


@app.post("/start")
def start():
    b = request.get_json()
    uid = clean(b["id"])
    d = os.path.join(ROOT, uid)
    name = b.get("name", "video.mp4")
    video = os.path.join(d, "video" + (os.path.splitext(name)[1] or ".mp4"))
    with open(video, "wb") as out:
        for p in sorted(x for x in os.listdir(d) if x.startswith("part")):
            fp = os.path.join(d, p)
            with open(fp, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(fp)
    JOBS[uid] = dict(state="queued", stage="Queued", progress=0.0, log=[],
                     base=os.path.splitext(name)[0])
    threading.Thread(target=run, args=(uid, video, b), daemon=True).start()
    return jsonify(job=uid)


@app.get("/status/<uid>")
def status(uid):
    j = JOBS.get(uid) or abort(404)
    return jsonify({k: j.get(k) for k in ("state", "stage", "progress", "log", "summary", "error")})


@app.get("/download/<uid>/<kind>")
def download(uid, kind):
    j = JOBS.get(uid)
    if not j or j["state"] != "done":
        abort(404)
    path, suffix = (j["xlsx"], "_transcript.xlsx") if kind == "xlsx" else (j["srt"], "_en.srt")
    return send_file(path, as_attachment=True, download_name=j["base"] + suffix)


def start_tunnel():
    exe = "/usr/local/bin/cloudflared"
    if not os.path.exists(exe):
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", exe)
        os.chmod(exe, 0o755)
    p = subprocess.Popen([exe, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            threading.Thread(target=lambda: [None for _ in p.stdout], daemon=True).start()
            return m.group(0)
    raise RuntimeError("Could not start tunnel")


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    threading.Thread(target=lambda: app.run(host="127.0.0.1", port=PORT, threaded=True), daemon=True).start()
    time.sleep(1)
    url = start_tunnel()
    print("\n" + "=" * 64 + "\nPaste this connection code into your website:\n\n  "
          + url + "#" + KEY + "\n\nKeep this cell running while you use the site.\n" + "=" * 64, flush=True)
    while True:
        time.sleep(60)
