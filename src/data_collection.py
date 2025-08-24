from pathlib import Path
from sec_edgar_downloader import Downloader

COMPANY_NAME = "max & xp"
EMAIL        = "zhaizhaoyue520@gmail.com"
OUT_DIR      = "data/raw_reports"

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK-A","TSLA","JPM","JNJ"]

def download_top10():
    dl = Downloader(COMPANY_NAME, EMAIL, OUT_DIR)
    for t in TICKERS:
        print(f"==> {t} 10-K (last 3)")
        dl.get("10-K", t, limit=2, download_details=True)
        print(f"==> {t} latest 10-Q (last 1)")
        dl.get("10-Q", t, limit=1,download_details=True)
    print("🎉 Top10 batch download finished")

if __name__ == "__main__":
    download_top10()