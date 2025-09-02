# 交互里先跑一段
from src.qa.answer_api import _facts_df
df = _facts_df()
print(type(df), (0 if df is None else len(df)))
print(df.columns.tolist()[:12] if df is not None else "NO DF")

#ython -m src.qa.testing/测试