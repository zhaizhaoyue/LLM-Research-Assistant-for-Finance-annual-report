
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.qa.hybrid_qa import answer_textual_or_mixed

def _hit(snippet, source_path="data/raw/US_AAPL_2023_10-K_x.html", page_no=12, chunk_id="cid1"):
    return {
        "snippet": snippet,
        "chunk_id": chunk_id,
        "meta": {
            "source_path": source_path,
            "page_no": page_no,
            "file_type": "text_chunk",
            "ticker": "AAPL",
            "fy": 2023,
        }
    }

def test_basic_rule_answer_no_llm():
    hits = [
        _hit("Total net sales decreased 3% or $11.0 billion during 2023 compared to 2022."),
        _hit("The weakness in foreign currencies accounted for more than the entire decrease.", page_no=13, chunk_id="cid2"),
    ]
    final, reasoning, cits = answer_textual_or_mixed(
        query="2023年净销售额同比变化？", hits=hits, filters={}, use_llm=False, k_ctx=4, max_chars=300
    )
    print("FINAL:", final)
    print("REASON:", reasoning)
    print("CITS :", cits)
    assert "decreased 3%" in final.lower()
    assert len(cits) >= 1

def test_empty_hits_graceful():
    final, reasoning, cits = answer_textual_or_mixed(
        query="What happened to net sales?", hits=[], filters={}, use_llm=False
    )
    print("FINAL(empty):", final)
    assert "无法给出充分的信息" in final  # 兜底提示
    assert cits == []

def test_messy_snippet_cleanup():
    messy = "  ●●  TOTAL  NET  SALES   \n  decreased   3%     \n\n during 2023  "
    final, _, _ = answer_textual_or_mixed(
        query="?", hits=[_hit(messy)], filters={}, use_llm=False
    )
    print("FINAL(clean):", final)
    assert "TOTAL NET SALES" in final or "total net sales" in final.lower()

if __name__ == "__main__":
    test_basic_rule_answer_no_llm()
    test_empty_hits_graceful()
    test_messy_snippet_cleanup()
    print("All tests ran.")
