import os
from datetime import date, timedelta
import pandas as pd
from langchain_core.tools import tool

from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

_data_path = os.path.join(os.path.dirname(__file__), 'data', 'coffee_sales.csv')
df = pd.read_csv(_data_path)

@tool
def top_n_revenue(n: int) -> str:
    """Return the top n coffee products by total revenue."""
    return df.groupby('coffee_name')['price'].sum().nlargest(n).to_json()

@tool
def top_n_item(n: int) -> str:
    """Return the top n coffee products by number of items sold."""
    return df.groupby('coffee_name').size().nlargest(n).to_json()

@tool
def save_report(history: list, file_name: str = "report.txt") -> str:
    """Save the session query history to file_name."""
    with open(file_name, "w") as f:
        for i, entry in enumerate(history, 1):
            f.write(f"Q{i}: {entry['query']}\n")
            f.write(f"Result: {entry['result']}\n\n")
    return f"Report saved with {len(history)} entries."
            
@tool
def load_csv(filepath: str = "") -> str:
    """Load a CSV file into the dataframe. Returns column names and row count.
    If filepath is omitted or cannot be found, loads the default coffee sales dataset."""
    global df
    candidates = []
    if filepath:
        candidates.append(filepath)
        candidates.append(os.path.join(os.path.dirname(_data_path), filepath))
    candidates.append(_data_path)

    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            note = f" (requested '{filepath}' not found, loaded default dataset instead)" if filepath and path == _data_path else ""
            return f"Loaded {len(df)} rows with columns: {list(df.columns)}{note}"

    return f"Could not find '{filepath}' or the default dataset."

@tool
def describe_data(column: str = "") -> str:
    """Describe the loaded dataframe. Pass a column name for specific stats, or empty string for full summary."""
    if column and column in df.columns:
        return df[column].describe().to_json()
    return df.describe().to_json()

@tool
def filter_rows(column: str, value: str) -> str:
    """Filter rows where column equals value. Returns filtered row count and preview."""
    filtered = df[df[column].astype(str) == value]
    return filtered.head(5).to_json(orient='records')

@tool
def plot_chart(column: str, chart_type: str = "bar") -> str:
    """Plot a chart of a column. chart_type can be 'bar' or 'pie'. Saves as chart.png."""
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    data = df.groupby(column)['price'].sum()
    data.plot(kind=chart_type, figsize=(10, 6), title=f"Revenue by {column}")
    plt.tight_layout()
    plt.savefig('chart.png')
    plt.close()
    return f"Chart saved as chart.png showing revenue by {column}"

tools = [load_csv, describe_data, filter_rows, top_n_revenue, top_n_item, plot_chart]

