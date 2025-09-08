#建索引（BM25 + 金融友好分词 + 预留向量字段）
'''
chmod +x scripts/init_index.sh
ES_URL=http://localhost:9200 ES_AUTH="elastic:password" INDEX_NAME=finance-chunks src/scripts/init_index.sh
'''

#!/usr/bin/env bash
set -euo pipefail

ES_URL=${ES_URL:-http://localhost:9200}
INDEX=${INDEX_NAME:-finance-chunks}

# 可选鉴权：export ES_AUTH="user:pass"
AUTH_FLAG=""
if [ -n "${ES_AUTH:-}" ]; then
  AUTH_FLAG="-u ${ES_AUTH}"
fi

# 删除旧索引
curl -sS ${AUTH_FLAG} -XDELETE "${ES_URL}/${INDEX}" >/dev/null || true

# 创建新索引（BM25 默认；自定义 analyzer 保留数字/货币/连字符/小数点）
curl -sS ${AUTH_FLAG} -H 'Content-Type: application/json' -XPUT "${ES_URL}/${INDEX}" -d '{
  "settings": {
    "index": {
      "number_of_shards": 1,
      "number_of_replicas": 0
    },
    "analysis": {
      "analyzer": {
        "finance_text": {
          "type": "custom",
          "tokenizer": "pattern",
          "filter": ["lowercase"],
          "pattern": "[A-Za-z0-9%$€¥£\\.-]+"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "chunk_id": { "type": "keyword" },
      "ticker":   { "type": "keyword" },
      "form":     { "type": "keyword" },        // 10-K / 10-Q
      "fy":       { "type": "integer" },
      "fq":       { "type": "keyword" },
      "accno":    { "type": "keyword" },
      "section":  { "type": "keyword" },        // e.g., Item 7, Note 13
      "heading":  { "type": "text", "analyzer": "finance_text", "fields": { "raw": { "type": "keyword" } } },
      "title":    { "type": "text", "analyzer": "finance_text", "fields": { "raw": { "type": "keyword" } } },
      "page_no":  { "type": "integer" },
      "content":  { "type": "text", "analyzer": "finance_text" },

      // 预留 dense 向量字段（ES 8.11+）
      "embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "dot_product"
      }
    }
  }
}'
echo
echo "Index ${INDEX} created."
