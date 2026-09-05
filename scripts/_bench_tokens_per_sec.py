#!/usr/bin/env python3
"""Benchmark vLLM throughput.

Measures, for a set of test profiles, with streaming enabled:
  - Time To First Token (TTFT, ms)
  - Inter-token latency (ms/token, generation phase)
  - Generation throughput (tokens/s, generation phase)
  - End-to-end throughput (tokens/s, including prefill)

Writes JSONL events to <out-dir>/events.jsonl and a human-readable
summary to <out-dir>/summary.txt.

Usage:
  python3 _bench_tokens_per_sec.py --out-dir <path> --port 8000 --model <id>
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone


def post_stream(url: str, body: dict, timeout: int = 600):
    """Yield (chunk_index, delta_dict, ts_recv) for each SSE line.

    Stops at the chunk that carries `usage` (end of stream) or at
    the `data: [DONE]` sentinel.
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    resp.read1 = resp.read1  # noqa: just to keep pyflakes quiet
    buf = b""
    idx = 0
    for raw in resp:
        buf += raw
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.decode("utf-8", "replace").rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            idx += 1
            yield idx, obj, time.monotonic()


def run_one(name: str, url: str, model: str, prompt: str, max_tokens: int,
            repeats: int, out_dir: str) -> dict:
    """Run a single profile `repeats` times, return aggregate stats."""
    body_base = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # deterministic for benchmarking
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    runs = []
    for r in range(repeats):
        body = dict(body_base)
        body["messages"] = [{"role": "user", "content": prompt}]
        t_start = time.monotonic()
        chunks = []
        t_first = None
        t_last = None
        usage = None
        for _idx, obj, t_recv in post_stream(f"{url}/v1/chat/completions", body):
            chunks.append(obj)
            if t_first is None:
                t_first = t_recv
            t_last = t_recv
            if obj.get("usage"):
                usage = obj["usage"]
        t_end = t_last if t_last else time.monotonic()

        if not usage:
            print(f"  [{name} run {r+1}/{repeats}] no usage chunk; aborting", file=sys.stderr)
            continue
        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok = usage.get("completion_tokens", 0)
        reasoning_tok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        total_tok = prompt_tok + comp_tok

        ttft_ms = (t_first - t_start) * 1000.0 if t_first else 0.0
        # Generation phase: from first token to end
        gen_s = max(t_end - (t_first or t_start), 1e-6)
        # End-to-end: from request start to last token
        e2e_s = max(t_end - t_start, 1e-6)

        gen_tps = comp_tok / gen_s if gen_s > 0 else 0.0
        e2e_tps = total_tok / e2e_s if e2e_s > 0 else 0.0
        prefill_tps = prompt_tok / max(t_first - t_start, 1e-6) if t_first else 0.0

        runs.append({
            "run": r + 1,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "reasoning_tokens": reasoning_tok,
            "ttft_ms": ttft_ms,
            "gen_s": gen_s,
            "e2e_s": e2e_s,
            "prefill_tps": prefill_tps,
            "gen_tps": gen_tps,
            "e2e_tps": e2e_tps,
        })
        with open(os.path.join(out_dir, "events.jsonl"), "a") as f:
            f.write(json.dumps({"profile": name, **runs[-1]}) + "\n")
        print(f"  [{name} run {r+1}/{repeats}] "
              f"prompt={prompt_tok} comp={comp_tok} (reason={reasoning_tok}) | "
              f"TTFT={ttft_ms:.0f}ms | gen={gen_tps:.1f} tok/s | e2e={e2e_tps:.1f} tok/s | prefill={prefill_tps:.0f} tok/s")
    if not runs:
        return {}
    return {
        "n": len(runs),
        "prompt_tokens": runs[0]["prompt_tokens"],
        "completion_tokens_mean": statistics.mean(r["completion_tokens"] for r in runs),
        "reasoning_tokens_mean": statistics.mean(r["reasoning_tokens"] for r in runs),
        "ttft_ms_mean": statistics.mean(r["ttft_ms"] for r in runs),
        "ttft_ms_p50": statistics.median(r["ttft_ms"] for r in runs),
        "gen_tps_mean": statistics.mean(r["gen_tps"] for r in runs),
        "gen_tps_p50": statistics.median(r["gen_tps"] for r in runs),
        "e2e_tps_mean": statistics.mean(r["e2e_tps"] for r in runs),
        "e2e_tps_p50": statistics.median(r["e2e_tps"] for r in runs),
        "prefill_tps_mean": statistics.mean(r["prefill_tps"] for r in runs),
        "prefill_tps_p50": statistics.median(r["prefill_tps"] for r in runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "events.jsonl"), "w").close()  # truncate

    url = f"http://127.0.0.1:{args.port}"
    print(f"[bench] target: {url}  model={args.model}  repeats={args.repeats}")
    print(f"[bench] out:   {args.out_dir}")
    print(f"[bench] time:  {datetime.now(timezone.utc).isoformat()}")
    print()

    # Profile 1: short prompt, short generation (general "feel" of the server).
    PROMPT_SHORT = "List 5 capitals of Europe, one per line."
    # Profile 2: medium prompt (~2K tokens) forcing thinking, generate 512.
    PROMPT_MED = (
        "You are a senior software architect. Review the following code for "
        "race conditions, deadlocks, and resource leaks. Be thorough. "
        "Code:\n"
        + ("def handler(req):\n    return process(req)\n" * 60)
        + "\nNow provide your review."
    )
    # Profile 3: long context fill (5K tokens), short generation.
    PROMPT_LONG = (
        "Below is a long document. When you reach the end, output the single "
        "line: ACK.\n\n"
        + ("The quick brown fox jumps over the lazy dog. " * 800)
    )

    profiles = [
        ("short_prompt_short_gen",  PROMPT_SHORT, 128),
        ("medium_prompt_medium_gen", PROMPT_MED,   512),
        ("long_prompt_short_gen",    PROMPT_LONG,  64),
    ]

    summary = {"target": url, "model": args.model, "repeats": args.repeats, "profiles": {}}
    for name, prompt, max_tokens in profiles:
        print(f"[bench] profile: {name}  (max_tokens={max_tokens})")
        agg = run_one(name, url, args.model, prompt, max_tokens, args.repeats, args.out_dir)
        summary["profiles"][name] = agg
        print()

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable summary
    lines = [
        "vLLM THROUGHPUT BENCHMARK",
        "=" * 60,
        f"target: {url}",
        f"model:  {args.model}",
        f"repeats: {args.repeats} per profile",
        "",
    ]
    for name, agg in summary["profiles"].items():
        if not agg:
            continue
        lines += [
            f"--- {name} ---",
            f"  prompt tokens (one req):  {agg['prompt_tokens']}",
            f"  completion tokens (mean): {agg['completion_tokens_mean']:.1f}  (reasoning: {agg['reasoning_tokens_mean']:.1f})",
            f"  TTFT:                     mean={agg['ttft_ms_mean']:.0f} ms  p50={agg['ttft_ms_p50']:.0f} ms",
            f"  Prefill throughput:       mean={agg['prefill_tps_mean']:.0f} tok/s  p50={agg['prefill_tps_p50']:.0f} tok/s",
            f"  Generation throughput:    mean={agg['gen_tps_mean']:.1f} tok/s  p50={agg['gen_tps_p50']:.1f} tok/s",
            f"  End-to-end throughput:    mean={agg['e2e_tps_mean']:.1f} tok/s  p50={agg['e2e_tps_p50']:.1f} tok/s",
            "",
        ]
    txt = "\n".join(lines)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
