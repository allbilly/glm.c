#!/usr/bin/env python3
import csv
import json
import os
import re
from datetime import datetime, timezone


def read_real(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("real "):
                try:
                    return float(line.split()[1])
                except Exception:
                    return None
    return None


def tps_from_time(path: str, tokens: int):
    real = read_real(path)
    if real is None or real <= 0:
        return None
    return tokens / real


def tps_from_llama_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    row = next((x for x in data if isinstance(x, dict) and x.get("n_gen", 0) > 0), None)
    if not row:
        return None
    try:
        return float(row["avg_ts"])
    except Exception:
        return None


def thread_sweep_results(tokens: int):
    out = []
    rx = re.compile(r"glm_tps_glm-cpu-omp-t(\d+)\.time\.txt$")
    tmp_dir = os.environ.get("GLM_TMP_DIR", os.path.join(os.getcwd(), "tmp"))
    if not os.path.isdir(tmp_dir):
        return out
    for name in sorted(os.listdir(tmp_dir)):
        m = rx.match(name)
        if not m:
            continue
        threads = int(m.group(1))
        path = os.path.join(tmp_dir, name)
        tps = tps_from_time(path, tokens)
        if tps is not None:
            out.append({"threads": threads, "tok_s": tps, "time_file": path})
    return out


def main():
    tokens = int(os.environ.get("BENCH_TOKENS", "16"))
    tmp_dir = os.environ.get("GLM_TMP_DIR", os.path.join(os.getcwd(), "tmp"))
    os.makedirs(tmp_dir, exist_ok=True)

    def tpath(name: str):
        return os.path.join(tmp_dir, name)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tokens": tokens,
        "results": [
            {
                "mode": "glm_cpu_decode",
                "tok_s": tps_from_time(tpath("glm_decode_cpu.time.txt"), tokens),
                "time_file": tpath("glm_decode_cpu.time.txt"),
            },
            {
                "mode": "glm_metal_decode",
                "tok_s": tps_from_time(tpath("glm_decode_metal.time.txt"), tokens),
                "time_file": tpath("glm_decode_metal.time.txt"),
            },
            {
                "mode": "glm_cpu_tps",
                "tok_s": tps_from_time(tpath("glm_tps_glm-cpu.time.txt"), tokens),
                "time_file": tpath("glm_tps_glm-cpu.time.txt"),
            },
            {
                "mode": "glm_metal_tps",
                "tok_s": tps_from_time(tpath("glm_tps_glm-metal.time.txt"), tokens),
                "time_file": tpath("glm_tps_glm-metal.time.txt"),
            },
            {
                "mode": "llama_cpu",
                "tok_s": tps_from_llama_json(tpath("glm_tps_llama-bench-cpu.json")),
                "json_file": tpath("glm_tps_llama-bench-cpu.json"),
            },
            {
                "mode": "llama_metal",
                "tok_s": tps_from_llama_json(tpath("glm_tps_llama-bench-metal.json")),
                "json_file": tpath("glm_tps_llama-bench-metal.json"),
            },
            {
                "mode": "glm_prefill_cpu_time_s",
                "tok_s": read_real(tpath("glm_prefill_cpu.time.txt")),
                "time_file": tpath("glm_prefill_cpu.time.txt"),
            },
            {
                "mode": "glm_prefill_metal_time_s",
                "tok_s": read_real(tpath("glm_prefill_metal.time.txt")),
                "time_file": tpath("glm_prefill_metal.time.txt"),
            },
            {
                "mode": "glm_matvec_proxy_metal",
                "tok_s": tps_from_time(
                    tpath("glm_matvec_proxy_metal.time.txt"), tokens
                ),
                "time_file": tpath("glm_matvec_proxy_metal.time.txt"),
            },
        ],
        "omp_thread_sweep": thread_sweep_results(tokens),
    }

    json_out = tpath("glm_bench_summary.json")
    csv_out = tpath("glm_bench_summary.csv")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "tok_s_or_seconds", "artifact"])
        for row in summary["results"]:
            artifact = row.get("time_file") or row.get("json_file") or ""
            value = row.get("tok_s")
            writer.writerow(
                [row["mode"], "" if value is None else f"{value:.6f}", artifact]
            )
        for row in summary["omp_thread_sweep"]:
            writer.writerow(
                [
                    f"omp_threads_{row['threads']}",
                    f"{row['tok_s']:.6f}",
                    row["time_file"],
                ]
            )

    print("wrote", json_out)
    print("wrote", csv_out)


if __name__ == "__main__":
    main()
