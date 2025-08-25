import argparse
from cleaning import text_clean, table_normalize

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    #clean
    p1 = subparsers.add_parser("text-clean")
    p1.add_argument("--input_file", required=True)
    p1.add_argument("--output_jsonl", required=True)
    p1.add_argument("--output_parquet", required=True)

    # table_normalize
    p2 = subparsers.add_parser("table-clean")
    p2.add_argument("--input", required=True)
    p2.add_argument("--metric_map", required=True)
    p2.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    if args.command == "text-clean":
        text_clean.main_from_args(args)
    elif args.command == "table-clean":
        table_normalize.main_from_args(args)

if __name__ == "__main__":
    main()
