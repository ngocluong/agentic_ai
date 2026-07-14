"""
plot_eval.py — reads eval_history_w6.csv and produces eval_trend.png
showing avg score over time + quality threshold + weakest question analysis.
"""
import csv
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict

HISTORY_FILE = "w6_rag_agent/eval_history_w6.csv"
OUTPUT_FILE  = "w6_rag_agent/eval_trend.png"
THRESHOLD    = 4.0

QUESTION_LABELS = {
    "q1":  "RAG: two phases",
    "q2":  "RAG: reduces hallucination",
    "q3":  "LangChain: StrOutputParser vs JsonOutputParser",
    "q4":  "LangChain: LCEL and chain execution",
    "q5":  "ChromaDB: PersistentClient vs EphemeralClient",
    "q6":  "ChromaDB: distance above 1.2",
    "q7":  "ReAct: three steps in loop",
    "q8":  "ReAct: max_steps purpose",
    "q9":  "LangGraph: node vs edge",
    "q10": "LangGraph: MemorySaver",
}

CATEGORIES = {
    "RAG":        ["q1", "q2"],
    "LangChain":  ["q3", "q4"],
    "ChromaDB":   ["q5", "q6"],
    "ReAct":      ["q7", "q8"],
    "LangGraph":  ["q9", "q10"],
}


def load_history(filepath: str) -> list[dict]:
  rows = []
  with open(filepath, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      row["date"] = datetime.strptime(row["date"], "%Y-%m-%d")
      row["overall_score"] = float(row["overall_score"])
      row["passed"] = int(row["passed"])
      for q in QUESTION_LABELS:
        if q in row:
          row[q] = int(row[q])
      rows.append(row)
  return rows


def avg_per_question(rows: list[dict]) -> dict:
  totals = defaultdict(list)
  for row in rows:
    for q in QUESTION_LABELS:
      if q in row:
        totals[q].append(row[q])
  return {q: sum(vals) / len(vals) for q, vals in totals.items() if vals}


def avg_per_category(q_avgs: dict) -> dict:
  result = {}
  for cat, questions in CATEGORIES.items():
    vals = [q_avgs[q] for q in questions if q in q_avgs]
    if vals:
      result[cat] = sum(vals) / len(vals)
  return result


def plot(rows: list[dict], q_avgs: dict, cat_avgs: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RAG Agent Evaluation — Accuracy Trend", fontsize=14, fontweight="bold")

    # ── Left: avg score over time ──────────────────────────────────────────
    ax1 = axes[0]
    dates  = [r["date"] for r in rows]
    scores = [r["overall_score"] for r in rows]

    ax1.plot(dates, scores, marker="o", linewidth=2, color="#2196F3", label="Avg score")
    ax1.axhline(THRESHOLD, color="#F44336", linestyle="--", linewidth=1.2,
                label=f"Quality threshold ({THRESHOLD})")
    ax1.fill_between(dates, scores, THRESHOLD,
                     where=[s >= THRESHOLD for s in scores],
                     alpha=0.1, color="#4CAF50", label="Above threshold")
    ax1.fill_between(dates, scores, THRESHOLD,
                     where=[s < THRESHOLD for s in scores],
                     alpha=0.1, color="#F44336", label="Below threshold")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Avg Score (out of 5)")
    ax1.set_ylim(0, 5.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_title("Score Over Time")

    # ── Right: per-category avg ────────────────────────────────────────────
    ax2 = axes[1]
    if cat_avgs:
        cats   = list(cat_avgs.keys())
        values = [cat_avgs[c] for c in cats]
        colors = ["#4CAF50" if v >= THRESHOLD else "#F44336" for v in values]
        bars   = ax2.barh(cats, values, color=colors, edgecolor="white", height=0.5)
        ax2.axvline(THRESHOLD, color="#333", linestyle="--", linewidth=1, label=f"Threshold {THRESHOLD}")
        for bar, val in zip(bars, values):
            ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}", va="center", fontsize=9)
        ax2.set_xlim(0, 5.5)
        ax2.set_xlabel("Avg Score (out of 5)")
        ax2.set_title("Avg Score by Category")
        ax2.legend(fontsize=8)
        ax2.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "No per-question data yet\n(run eval with q1-q10 columns)",
                 ha="center", va="center", transform=ax2.transAxes, color="gray")
        ax2.set_title("Avg Score by Category")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()
    print(f"Chart saved → {OUTPUT_FILE}")


def print_analysis(q_avgs: dict, cat_avgs: dict):
    print("\n=== PER-QUESTION AVERAGES ===")
    sorted_q = sorted(q_avgs.items(), key=lambda x: x[1])
    for q, avg in sorted_q:
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        label = QUESTION_LABELS.get(q, q)
        print(f"  {q:3s} {bar} {avg:.1f}  {label}")

    if cat_avgs:
        print("\n=== CATEGORY AVERAGES (worst → best) ===")
        sorted_cats = sorted(cat_avgs.items(), key=lambda x: x[1])
        for cat, avg in sorted_cats:
            flag = " ← weakest" if cat == sorted_cats[0][0] else ""
            print(f"  {avg:.1f}/5  {cat}{flag}")

        worst_cat = sorted_cats[0][0]
        print(f"\n⚠️  Weakest category: '{worst_cat}' — focus prompt tuning here next")


if __name__ == "__main__":
  if not os.path.isfile(HISTORY_FILE):
    print(f"'{HISTORY_FILE}' not found. Run evaluator.py first.")
    exit(1)

  rows = load_history(HISTORY_FILE)
  print(f"Loaded {len(rows)} eval runs from {HISTORY_FILE}")

  q_avgs   = avg_per_question(rows)
  cat_avgs = avg_per_category(q_avgs)

  plot(rows, q_avgs, cat_avgs)
  print_analysis(q_avgs, cat_avgs)