import os
from dotenv import load_dotenv
from src.pipeline import run_pipeline


def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("Missing GOOGLE_API_KEY or TAVILY_API_KEY in .env")
        return

    ticker = "VND=X"
    print(f"Starting AI Forecasting Pipeline for {ticker}")
    run_pipeline(ticker)


if __name__ == "__main__":
    main()
