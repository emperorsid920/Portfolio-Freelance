# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from Stock import Portfolio

if __name__ == '__main__':
    portfolio = Portfolio()

    # Add stocks
    portfolio.add_stock("AAPL", 10, 150.0, "2024-12-01")
    portfolio.add_stock("GOOGL", 5, 2800.0, "2024-12-01")
    portfolio.add_stock("TSLA", 4, 3000, "2024-11-30")
    portfolio.add_stock("MSFT", 7, 700, "2024-09-25")

    # Print portfolio details
    print("Portfolio Details:")
    for ticker, details in portfolio.stocks.items():
        print(f"{ticker}: {details}")

    # Remove a stock
    portfolio.remove_stock("GOOGL")

    # Update current prices
    portfolio.update_current_price("AAPL", 160.0)
    portfolio.update_current_price("TSLA", 3100.0)
    portfolio.update_current_price("MSFT", 750.0)

    # Display updated portfolio
    print("\nPortfolio Details after updates:")
    for ticker, details in portfolio.stocks.items():
        print(f"{ticker}: {details}")

    # Calculate metrics
    portfolio.calculate_metrics()
