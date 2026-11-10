"""Send the two snapshots to a running llama-server and report answers and timings.

Usage: python run_demo.py <label> [snapshot names, comma-separated or "all"] [rules file]
The label (e.g. "thinking_off" or "thinking_on") is only written in the output file
name; thinking is set when the server starts. Without names (or with "all"), all
snapshots are sent. The rules file defaults to rules.txt.
Each run is saved in a new file with date and time, so no result is overwritten.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
URL = "http://127.0.0.1:8080/v1/chat/completions"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
ALL_SNAPSHOTS = ["snapshot_ok", "snapshot_error", "snapshot_error_noalert"]
SNAPSHOTS = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else ALL_SNAPSHOTS
RULES_FILE = sys.argv[3] if len(sys.argv) > 3 else "rules.txt"

# Instruction and rules form a fixed prefix, so the server can reuse it (prompt cache).
system_text = (HERE / "instruction.txt").read_text() + "\n" + (HERE / RULES_FILE).read_text()

results = {"rules_file": RULES_FILE}
for name in SNAPSHOTS:
    body = {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": (HERE / f"{name}.txt").read_text()},
        ],
        "temperature": 0,      # deterministic, so the run can be repeated
        "max_tokens": 4096,    # upper bound; with thinking on the answer may be long
    }
    request = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read())
    wall_s = time.perf_counter() - start

    message = data["choices"][0]["message"]
    timings = data.get("timings", {})
    results[name] = {"wall_s": round(wall_s, 2), "timings": timings,
                     "answer": message.get("content", ""),
                     "reasoning": message.get("reasoning_content", "")}

    print(f"===== {name} ({LABEL}, {RULES_FILE}) =====")
    print(f"wall time: {wall_s:.2f} s")
    for key in ["cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms"]:
        if key in timings:
            print(f"{key}: {timings[key]}")
    print(f"reasoning: {len(results[name]['reasoning'])} characters")
    print("answer:")
    print(results[name]["answer"])
    print()

out = HERE / f"results_{LABEL}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps(results, indent=2))
print(f"saved {out.name}")
