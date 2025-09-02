# src/qa/testing/run_utils_detect_demos.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.qa import utils_detect

def demo_types():
    print("=== DEMO A: detect_query_type ===")
    q1 = "What is the year-over-year revenue in 2023?"
    q2 = "Why did revenue decrease in 2023?"
    q3 = "Explain the drivers of cash increase"
    for q in (q1,q2,q3):
        print(q, "->", utils_detect.detect_query_type(q))

def demo_prev_quarter():
    print("\n=== DEMO B: locate_previous_period (QoQ exact quarter-ends) ===")
    cur = {"fy": 2023, "fq": "Q1", "period_end": "2023-12-31"}
    prev = utils_detect.locate_previous_period(cur, prefer="QoQ")
    print("Current:", cur, "-> Previous:", prev)

    cur = {"fy": 2024, "fq": "Q2", "period_end": "2024-03-31"}
    prev = utils_detect.locate_previous_period(cur, prefer="QoQ")
    print("Current:", cur, "-> Previous:", prev)

def demo_expand_alias():
    print("\n=== DEMO C: expand_with_alias ===")
    q = "AAPL 2023 revenue YoY"
    exp = utils_detect.expand_with_alias(q)
    print(exp)

if __name__ == "__main__":
    demo_types()
    demo_prev_quarter()
    demo_expand_alias()
    print("\n[OK] utils_detect demos finished.")
