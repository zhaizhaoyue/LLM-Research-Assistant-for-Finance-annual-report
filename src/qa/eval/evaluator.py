# src/qa/eval/evaluator.py
import json, csv, time
from pathlib import Path
from typing import List, Dict, Any
from src.qa.answer_api import answer_question
from .metrics import (
    metric_hit_at_k, metric_em_numeric, metric_reasonable_rate
)

def run_eval(eval_path: Path, out_answers: Path, out_report: Path):
    rows_answers = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            res = answer_question(q["query"], q.get("filters", {}))
            rows_answers.append({
                "id": q["id"],
                "type": q["type"],
                "query": q["query"],
                "expect": q.get("expect"),
                "answer": res["final_answer"],
                "used_method": res["used_method"],
                "latency_ms": res["latency_ms"],
                "citations": res["citations"],
            })

    # 写答案文件（留档）
    with out_answers.open("w", encoding="utf-8") as f:
        for r in rows_answers:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总指标
    report = aggregate_metrics(rows_answers)
    fieldnames = list(report.keys())
    with out_report.open("w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(report)

def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 这里示例：你可在 metrics.py 里实现更细
    hitk = metric_hit_at_k(rows)                   # 需要在 answer 侧回传命中排名（可由 retrieval 暴露）
    em, num_err = metric_em_numeric(rows)
    reason_rate = metric_reasonable_rate(rows)     # 简单基于关键字/规则
    avg_lat = sum(r["latency_ms"] for r in rows)/max(1,len(rows))
    return {
        "Hit@K": round(hitk, 4),
        "ExactMatch_or_NumOK": round(em, 4),
        "MeanNumError": round(num_err, 4),
        "ReasonableRate": round(reason_rate, 4),
        "AvgLatencyMs": int(avg_lat),
        "N": len(rows),
    }
