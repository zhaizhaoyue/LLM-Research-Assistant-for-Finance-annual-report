# AIE数值检索系统

## 概述

AIE (Automated Information Extraction) 数值检索系统是一个专门为财务数据数值检索任务设计的端到端处理流水线。该系统已成功整合到现有的多模态LLM金融研究助手中，复用了现有的RAG检索系统和数据基础设施。

## 🎯 核心功能

- **数据源整合**: 统一处理 `data/chunked/`、`data/processed/` 和 `data/compact_tables/` 中的财务数据
- **智能检索**: 复用现有的混合检索系统 (BM25 + Dense + Cross-Encoder)，针对数值查询进行优化
- **精确提取**: 支持数值、货币、百分比等格式的精确识别和标准化
- **流水线处理**: 分段 → 检索 → 摘要 → 提取的完整处理流程

## 📁 文件结构

```
src/aie_for_numeric_retrieval/
├── pipeline.py                 # 主流水线，整合所有组件
├── financial_data_adapter.py   # 数据适配器，统一数据源接口
├── retrieval.py               # 检索模块，整合现有RAG系统
├── extraction.py              # 数值提取模块
├── summarization.py           # 文档摘要模块
├── segmentation.py            # 文档分段模块
├── models/
│   └── llm_interface.py       # LLM接口
├── demo_numeric_retrieval.py  # 演示脚本
├── test_integration.py        # 整合测试
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 环境准备

确保现有环境已配置：
- Python 3.8+
- 现有的依赖包已安装
- CUDA (可选，用于GPU加速)

### 2. 数据准备

确保数据目录结构正确：
```bash
data/
├── chunked/        # 文本块数据 (主要数据源)
├── processed/      # 结构化数据 (备选)
├── compact_tables/ # 表格数据 (备选)
└── index/          # FAISS索引
```

### 3. 构建索引

如果索引不存在，运行：
```bash
python src/chunking_and_embedding/embedding.py
```

### 4. 配置LLM

设置DeepSeek API Key：
```bash
export DEEPSEEK_API_KEY="your-api-key"
```

### 5. 运行测试

验证整合是否成功：
```bash
python src/aie_for_numeric_retrieval/test_integration.py
```

### 6. 运行演示

```bash
python src/aie_for_numeric_retrieval/demo_numeric_retrieval.py
```

## 💻 使用示例

```python
from src.aie_for_numeric_retrieval.pipeline import AIEPipeline
from src.aie_for_numeric_retrieval.extraction import ExtractionTarget
from src.aie_for_numeric_retrieval.models.llm_interface import LLMInterface

# 配置
config = {
    "stages": {"segmentation": True, "retrieval": True, "summarization": True, "extraction": True},
    "retrieval": {"index_dir": "data/index", "content_dir": "data/chunked"},
    "extraction": {"extraction_method": "hybrid"}
}

llm_config = {"provider": "deepseek", "model_name": "deepseek-chat"}
llm_interface = LLMInterface(llm_config)

# 初始化流水线
pipeline = AIEPipeline(config, llm_interface)

# 定义提取目标
targets = [
    ExtractionTarget("revenue_2023", "2023年营业收入", "number", unit="美元"),
    ExtractionTarget("net_income_2023", "2023年净利润", "number", unit="美元")
]

# 处理财务文档
result = pipeline.process_financial_document(
    ticker="AAPL",
    year=2023,
    query="What was Apple's revenue and net income in 2023?",
    extraction_targets=targets,
    form_type="10-K"
)

# 查看结果
for extraction in result.extractions:
    print(f"{extraction.target.name}: {extraction.value}")
    print(f"置信度: {extraction.confidence:.3f}")
```

## 🔧 配置选项

主要配置文件: `configs/aie_numeric_config.yaml`

关键配置项：
- `retrieval.index_dir`: 索引目录
- `retrieval.content_dir`: 内容目录  
- `retrieval.model`: 嵌入模型
- `llm.provider`: LLM提供商 (deepseek/openai/huggingface)
- `extraction.method`: 提取方法 (llm/regex/hybrid)

## 📊 性能特点

- **数据源优化**: 优先使用 `data/chunked/` 获得最佳上下文
- **检索优化**: 针对数值查询的关键词增强
- **提取准确性**: 混合策略 + 置信度评估
- **处理速度**: GPU加速 + 缓存机制

## 🔍 整合特点

### 与现有系统的整合

1. **复用RAG检索器**: 直接使用 `src/rag/retriever/` 中的混合检索系统
2. **复用数据基础设施**: 使用现有的 `data/` 目录结构
3. **复用索引系统**: 使用 `src/chunking_and_embedding/` 构建的FAISS索引
4. **无缝集成**: 不修改现有代码，只添加新功能

### 数值检索优化

1. **查询增强**: 自动添加数值相关关键词
2. **结果重排**: 基于数值密度的重新排序
3. **精确提取**: 支持货币、百分比、普通数值格式
4. **置信度评估**: 多维度置信度计算

## 🧪 测试与验证

运行完整测试套件：
```bash
# 整合测试
python src/aie_for_numeric_retrieval/test_integration.py

# 演示测试
python src/aie_for_numeric_retrieval/demo_numeric_retrieval.py
```

测试覆盖：
- ✅ 模块导入
- ✅ 数据适配器
- ✅ 检索系统整合
- ✅ 提取目标创建
- ✅ 流水线配置
- ✅ 目录结构

## 📝 注意事项

1. **数据依赖**: 需要现有的财务数据和索引
2. **API密钥**: 需要配置LLM API密钥
3. **资源要求**: GPU可选但推荐用于加速
4. **兼容性**: 与现有系统完全兼容，不影响原功能

## 🔗 相关文档

- [集成指南](../../docs/aie_integration_guide.md) - 详细的整合说明
- [配置文件](../../configs/aie_numeric_config.yaml) - 完整配置选项
- [原始RAG系统](../rag/retriever/) - 底层检索系统
- [数据处理](../chunking_and_embedding/) - 数据预处理

## 🎯 适用场景

- 财务报告数值提取
- 投资研究数据分析
- 监管合规数据检索
- 财务指标对比分析
- 自动化财务报告生成

## 📞 技术支持

如遇问题，请检查：
1. 数据目录是否完整
2. 索引是否已构建
3. API密钥是否正确配置
4. 依赖包是否完整安装

---

*AIE数值检索系统 - 专为财务数据数值检索优化的智能处理流水线*