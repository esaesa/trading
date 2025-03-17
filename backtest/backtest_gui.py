import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import json
import os

CONFIG_PATH = "config.json"

class BacktestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Backtesting GUI")
        self.root.geometry("500x400")
        
        self.load_config_button = tk.Button(root, text="Load Config", command=self.load_config)
        self.load_config_button.pack(pady=10)
        
        self.run_backtest_button = tk.Button(root, text="Run Backtest", command=self.run_backtest)
        self.run_backtest_button.pack(pady=10)
        
        self.show_results_button = tk.Button(root, text="Show Results", command=self.show_results)
        self.show_results_button.pack(pady=10)
        
        self.log_text = tk.Text(root, height=15, width=60)
        self.log_text.pack(pady=10)

    def load_config(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filepath:
            with open(filepath, 'r') as file:
                config_data = json.load(file)
            messagebox.showinfo("Config Loaded", f"Config file loaded: {filepath}")
            self.log_message(f"Loaded config: {json.dumps(config_data, indent=4)}")

    def run_backtest(self):
        self.log_message("Running backtest...")
        try:
            subprocess.run(["python", "runner.py"], check=True)
            self.log_message("Backtest completed successfully.")
        except subprocess.CalledProcessError as e:
            self.log_message(f"Error running backtest: {e}")
            messagebox.showerror("Backtest Error", "An error occurred while running the backtest.")
    
    def show_results(self):
        self.log_message("Opening backtest results...")
        # data_dir = os.path.dirname(__file__)
        results_file =data_dir = os.path.dirname(__file__) + "/backtest_results/" + "backtest_results.html"
        if os.path.exists(results_file):
            os.system(f"start {results_file}" if os.name == "nt" else f"open {results_file}")
        else:
            messagebox.showerror("Error", "Results file not found.")
    
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    gui = BacktestGUI(root)
    root.mainloop()
