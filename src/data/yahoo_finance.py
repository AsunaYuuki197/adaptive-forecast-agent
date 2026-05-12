import yfinance as yf
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_ticker_data(ticker: str, period="6mo"):
    logger.info(f"Fetching data for {ticker} from Yahoo Finance...")
    data = yf.Ticker(ticker).history(period=period)

    data.reset_index(inplace=True)
    data = data.dropna()
    data.drop_duplicates(inplace=True)

    return data
