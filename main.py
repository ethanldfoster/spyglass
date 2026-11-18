import datetime
import pandas as pd
from pathlib import Path

from utils import (
    get_historical_data,
    get_market_caps,
    calc_ceiling_performance,
    calc_floor_performance,
    get_sigma_aug,
)



if Path("inputs/historical_data.csv").is_file():
    historical_data = pd.read_csv("inputs/historical_data.csv")
else:
    



if __name__ == "__main__":


    optimal_performance, _ = calc_ceiling_performance(sigma_aug)
    naive_performance = calc_floor_performance(market_caps, sigma_aug, N)
        
    print(f"Optimal performance TE: {optimal_performance:f}")
    print(f"Naive performance TE: {naive_performance:f}")
