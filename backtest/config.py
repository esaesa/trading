#config.py
# General Settings
from datetime import timedelta
import json
import os

import numpy as np

def load_config(config_path):
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, config_path)
    # config = load_config(config_path)
    with open(config_path, 'r') as file:
        return json.load(file)

config = load_config('config.json')



backtest_params = config['backtest_params']
strategy_params = config['strategy_params']
optimization_params = config['optimization_params']
symbols = backtest_params.get("symbols", ["BTCUSDT"])
timeframes = backtest_params.get("timeframes", ["1m"])
data_folder = backtest_params.get("data_folder", "../download/data")




# Optimization Functions
def maximize_func(stats):
    # print("Original duration:", stats['Max. Drawdown Duration'])
    
    """Custom function to maximize."""
    def convert_to_hours(duration):
        if isinstance(duration, timedelta):
            return duration.total_seconds() / 3600  # Convert to hours
        elif isinstance(duration, str):  # Parse string duration
            days, time = duration.split(' days ')
            hours, minutes, seconds = map(int, time.split(':'))
            return int(days) * 24 + hours + minutes / 60 + seconds / 3600
        else:
            return float(duration)  # Assume numerical
        
    def convert_to_days(duration):
        # print("Type of duration:", type(duration))  # Debug print
        if isinstance(duration, timedelta):
            return duration.total_seconds() / 60/60/24  # Convert to hours
        elif isinstance(duration, str):  # Parse string duration
            days, time = duration.split(' days ')
            hours, minutes, seconds = map(int, time.split(':'))
            return int(days)  + hours /24 + minutes / 1440 + seconds / 86400
        else:
            return float(duration)  # Assume numerical

    darwdown_days = convert_to_days(stats['Max. Drawdown Duration'])
    
    
    calmar_weight = 0.7  # 70% weight to Calmar
    alpha_weight = 0.3   # 30% weight to Alpha
    
    # Get metrics from last optimization window
    calmar = stats['Calmar Ratio']
    alpha = stats['Alpha [%]']
    

    
    # SAMBO prefers scores in [-1, 1] for default kernel configurations
    norm_calmar = np.clip(calmar / 3, -1.0, 1.0)  # Calmar ∈ [-3, 3] → [-1, 1]
    norm_alpha = np.clip(alpha / 300, -1.0, 1.0)  # Alpha ∈ [-300%, 300%] → [-1, 1]
    
    # SAMBO works best with non-zero gradients
    score = calmar_weight * norm_calmar + alpha_weight * norm_alpha
    
    # SAMBO-specific penalty for drawdowns
    if stats["Max. Drawdown [%]"] < -25 or darwdown_days > 48:
        score -= 0.5  # Maintains gradient continuity vs multiplicative penalties
    
    return np.clip(score, -1.0, 1.0)  # Final clamp
    # print("Converted to days:", converted_value)
    # print("Max Drawdown [%]:", -stats['Max. Drawdown [%]'])
    # if (stats['Max. Drawdown [%]'] < -30):
        # print("Max. Drawdown Duration/24: ",stats['Max. Drawdown Duration'] ," days ", convert_to_hours(stats['Max. Drawdown Duration'])/24)
        # print("Max Drawdown: ", stats['Max. Drawdown [%]'])
        # return 0
    # else:
        # print("Max. Drawdown Duration/24 Good: ", convert_to_hours(stats['Max. Drawdown Duration'])/24)
        # print("CAGR: ", stats['CAGR [%]'])
        # Return the inverse of drawdown duration so that shorter durations (better performance) yield higher values.
        # return (stats['CAGR [%]']*1) / (convert_to_hours(stats['Max. Drawdown Duration'])*8)
   
    if ( darwdown_days > 34):
        return 0
    
    return stats['Calmar Ratio'] + stats['Alpha [%]']
    # return stats['Alpha [%]']
    # return stats['Beta']
        
    # return stats['Return [%]']

    # return stats['Return [%]'] * (100/(-stats['Max. Drawdown [%]']))
    # return stats['Return [%]'] * (100/(-stats['Max. Drawdown [%]'])) / darwdown_days
    # return stats['Return [%]'] / darwdown_days
    
    # print(-stats['Max. Drawdown [%]'])
    # print(stats['Max. Drawdown Duration'])
    # print (convert_to_days(stats['Max. Drawdown Duration']))
    # return converted_value
    # return 1/convert_to_hours(stats['Max. Drawdown Duration'])
    # return stats['CAGR [%]']
    # return stats['# Trades']
    # print(stats['Sharpe Ratio'])
    # return stats['Sharpe Ratio']
    # if (stats['Max. Drawdown [%]'] < -65 or
    #     convert_to_hours(stats['Max. Drawdown Duration'])/24 > 65 ) : 
    #     # print("Max. Drawdown Duration/24: ",stats['Max. Drawdown Duration'] ," days ", convert_to_hours(stats['Max. Drawdown Duration'])/24)
    #     # print("Max Drawdown: ", stats['Max. Drawdown [%]'])
    #     return -1
    # else:
        # print("Max. Drawdown Duration/24 Good: ", convert_to_hours(stats['Max. Drawdown Duration'])/24)
        # print("CAGR: ", stats['CAGR [%]'])
        # Return the inverse of drawdown duration so that shorter durations (better performance) yield higher values.
        # return (stats['CAGR [%]']*1) / (convert_to_hours(stats['Max. Drawdown Duration'])*8)
    
    # if abs(stats['Max. Drawdown [%]']) < 1e-10:
    #     return -float("inf")  # Or some other fallback value
    # if abs(stats['Max. Drawdown [%]']) < 1e-10:
    #     return -float("inf")  # Or some other fallback value
    # return (
    #     stats['CAGR [%]'] / abs(stats['Max. Drawdown [%]']) /
    #     (convert_to_hours(stats['Max. Drawdown Duration']) *10)
    # )
    # return 1/abs(stats['Max. Drawdown [%]'])
    # return stats["Max. Drawdown [%]"] + stats["Sharpe Ratio"]
    # return stats['CAGR [%]']/convert_to_hours(stats['Max. Drawdown Duration'])
    # return stats['CAGR [%]']

def constraint_func(data):
    """Constraint for optimization."""
    #Examle return data.so_size_multiplier > data.price_multiplier
    return True
