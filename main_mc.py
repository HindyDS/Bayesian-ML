import yfinance as yf
from monte_carlo import MonteCarlo


if __name__ == '__main__':
    # get historical market data
    btc = yf.Ticker("BTC")
    hist = btc.history(period="max")

    # Calculate historical return
    ret = hist['Close'].pct_change()

    # Instantiate our MonteCarlo class
    n_sim = 10
    mc = MonteCarlo(ret)
    results = mc.sample_from_data(n_sim)
