"""第1回 宿題（発展）: 応答時間とトークン数を計測し、モデル・質問ごとに比較する。

使い方:
    uv run --env-file .env python -m scripts.benchmark            # 本計測（モデル×質問×回数）
    uv run --env-file .env python -m scripts.benchmark --burst    # 間隔なしの連続リクエストでレート制限を確かめる
    uv run --env-file .env python -m scripts.benchmark --models gemini-3.8-flash --repeats 1

結果は results/ に CSV で保存し、集計（平均・最小・最大）を表示する。
"""

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.config import get_api_key

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "data" / "glossary.json"
RESULTS = ROOT / "results"
TIMEOUT_S = 120  # 1リクエストのタイムアウト（秒）

DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
FIELDS = [
    "timestamp", "model", "question", "trial", "status", "elapsed_s",
    "prompt_tokens", "output_tokens", "thoughts_tokens", "output_tok_per_s", "error",
]


def questions() -> dict[str, str]:
    """比較する質問。short=短い出力、long_out=長い出力、long_in=長い入力。"""
    glossary = GLOSSARY.read_text(encoding="utf-8")
    return {
        "short": "LoRAとは何ですか？一文で答えてください。",
        "long_out": "LoRAの仕組みを、数式・利点・欠点・使いどころを含めて詳しく説明してください。",
        "long_in": f"次の用語集を読み、第1回に関係する用語を3つ挙げてください。\n\n{glossary}",
    }


def measure(client: genai.Client, model: str, qid: str, prompt: str, trial: int) -> dict:
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "model": model, "question": qid, "trial": trial}
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except errors.APIError as e:
        row.update(status=e.code, elapsed_s=round(time.perf_counter() - start, 3), error=e.message)
        return row
    except Exception as e:  # タイムアウトなど通信エラー
        row.update(status="timeout" if "timeout" in type(e).__name__.lower() else "error",
                   elapsed_s=round(time.perf_counter() - start, 3), error=f"{type(e).__name__}: {e}")
        return row
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata
    output = (usage.candidates_token_count or 0) if usage else 0
    row.update(
        status=200,
        elapsed_s=round(elapsed, 3),
        prompt_tokens=usage.prompt_token_count if usage else None,
        output_tokens=output,
        thoughts_tokens=(usage.thoughts_token_count or 0) if usage else 0,
        output_tok_per_s=round(output / elapsed, 1) if elapsed > 0 else None,
    )
    return row


def run_with_retry(client, model, qid, prompt, trial, writer, retries=2, wait=60) -> dict:
    """429 は記録したうえで待って再試行する。"""
    for attempt in range(retries + 1):
        row = measure(client, model, qid, prompt, trial)
        writer.writerow(row)
        print(f"  {model:24} {qid:9} #{trial} status={row['status']} "
              f"{row['elapsed_s']}s out={row.get('output_tokens')} think={row.get('thoughts_tokens')}", flush=True)
        if row["status"] != 429 or attempt == retries:
            return row
        print(f"  429: {wait}秒待って再試行します")
        time.sleep(wait)
    return row


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if r["status"] == 200]
    groups: dict[tuple, list[dict]] = {}
    for r in ok:
        groups.setdefault((r["model"], r["question"]), []).append(r)

    print("\n| model | question | n | 時間 平均 (最小〜最大) s | 入力 | 出力 | 思考 | 出力tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    for (model, qid), rs in groups.items():
        t = [r["elapsed_s"] for r in rs]
        avg = lambda k: round(statistics.mean(r[k] for r in rs))  # noqa: E731
        print(f"| {model} | {qid} | {len(rs)} | {statistics.mean(t):.2f} ({min(t):.2f}〜{max(t):.2f}) "
              f"| {avg('prompt_tokens')} | {avg('output_tokens')} | {avg('thoughts_tokens')} "
              f"| {statistics.mean(r['output_tok_per_s'] for r in rs):.1f} |")
    errs = [r for r in rows if r["status"] != 200]
    print(f"\n成功 {len(ok)} 件 / エラー {len(errs)} 件 " +
          str({s: sum(1 for r in errs if r['status'] == s) for s in {r['status'] for r in errs}}))


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--questions", nargs="+", default=list(questions()))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--interval", type=float, default=6.0, help="リクエスト間隔（秒）")
    parser.add_argument("--burst", action="store_true", help="間隔なしで連続リクエスト（再試行しない）")
    parser.add_argument("--burst-count", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    client = genai.Client(
        api_key=get_api_key(), http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000)
    )
    qs = questions()
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or RESULTS / f"{'burst' if args.burst else 'benchmark'}-{stamp}.csv"

    rows = []
    with out.open("w", newline="", encoding="utf-8", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        if args.burst:
            model = args.models[0]
            for i in range(1, args.burst_count + 1):
                rows.append(run_with_retry(client, model, "short", qs["short"], i, writer, retries=0))
        else:
            first = True
            for model in args.models:
                for qid in args.questions:
                    for trial in range(1, args.repeats + 1):
                        if not first:
                            time.sleep(args.interval)
                        first = False
                        rows.append(run_with_retry(client, model, qid, qs[qid], trial, writer))

    print(f"\n保存しました: {out.relative_to(ROOT)}")
    summarize(rows)


if __name__ == "__main__":
    main()
