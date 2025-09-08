#导入数据（递归导入 chunked JSONL）
from __future__ import annotations
from pathlib import Path
from typing import Iterable
# -*- coding: utf-8 -*-
"""
递归遍历 chunked 目录，把每个 *.jsonl 中的一行当作一个 chunk 文档写入 ES。
只索引我们需要的字段；缺失字段会被安全跳过。
"""
'''
保持es-dev容器常驻
docker rm -f es-dev 2>$null

docker run -d --name es-dev -p 9200:9200 `
  -e discovery.type=single-node `
  -e xpack.security.enabled=false `
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" `
  docker.elastic.co/elasticsearch/elasticsearch:8.14.3

'''
'''
python -m src.scripts.tools.bulk_index_es `
  --es http://localhost:9200 `
  --index finance-chunks `
  --content-root data/chunked

'''
'''建立最小映射

$body = @'
{
  "mappings": {
    "properties": {
      "chunk_id": { "type": "keyword" },
      "content":  { "type": "text" }
    }
  }
}
'@
Invoke-RestMethod -Method Put -Uri http://localhost:9200/finance-chunks `
  -ContentType 'application/json' -Body $body

'''

import argparse, json
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Generator
from elasticsearch import Elasticsearch, helpers

KEEP_FIELDS = {
    "chunk_id", "id", "ticker", "form", "fy", "fq", "accno",
    "section", "heading", "title", "page_no", "content", "text", "raw_text", "embedding"
}

def normalize_doc(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 仅保留常用字段名并做映射
    chunk_id = obj.get("chunk_id") or obj.get("id") or obj.get("chunkId")
    content = obj.get("content") or obj.get("text") or obj.get("raw_text")
    if not chunk_id or not content:
        return None

    doc = {
        "chunk_id": chunk_id,
        "ticker": obj.get("ticker"),
        "form": obj.get("form"),
        "fy": obj.get("fy") or obj.get("year"),
        "fq": obj.get("fq"),
        "accno": obj.get("accno"),
        "section": obj.get("section"),
        "heading": obj.get("heading"),
        "title": obj.get("title"),
        "page_no": obj.get("page_no"),
        "content": content
    }
    # 如已提前计算好 embedding，可带上（长度需与 mapping dims 一致）
    if "embedding" in obj and isinstance(obj["embedding"], list):
        doc["embedding"] = obj["embedding"]
    return doc



def iter_jsonl_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.name == "text_chunks.jsonl":
        yield root
        return
    # 遍历目录及子目录下所有 text_chunks.jsonl 文件
    for p in root.rglob("text_chunks.jsonl"):
        yield p


def gen_actions(root: Path, index: str) -> Generator[Dict[str, Any], None, None]:
    for fp in iter_jsonl_files(root):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # 限制只处理常见字段，避免脏行
                obj = {k: v for k, v in obj.items() if k in KEEP_FIELDS or k in ("meta",)}
                doc = normalize_doc(obj if "meta" not in obj else {**obj, **(obj.get("meta") or {})})
                if not doc:
                    continue
                yield {
                    "_index": index,
                    "_id": doc["chunk_id"],
                    "_source": doc
                }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", default="http://localhost:9200")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--index", default="finance-chunks")
    ap.add_argument("--content-root", required=True, help="chunked 目录或单个 jsonl 文件")
    ap.add_argument("--batch-size", type=int, default=2000)
    args = ap.parse_args()

    es = Elasticsearch(args.es, basic_auth=(args.user, args.password) if args.user else None)
    root = Path(args.content_root)

    actions = gen_actions(root, args.index)
    es_bulk = es.options(request_timeout=120)
    helpers.bulk(es_bulk, actions, chunk_size=args.batch_size)


if __name__ == "__main__":
    main()
