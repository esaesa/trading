# download_gui.py
from datetime import datetime
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry  # Install tkcalendar using pip if not already installed
import threading
import os
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / 'config.json'

# No more TextLoggerHandler or configure_gui_logger
# (Remove or comment them out)

def load_config(config_path=CONFIG_PATH):
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    else:
        cfg = {}

    dl = cfg.setdefault('download', {})
    # --- Migration / defaults ---
    # symbols may be missing or stored as a CSV string or at root
    syms = dl.get('symbols', cfg.get('symbols', []))
    if isinstance(syms, str):
        syms = [s.strip() for s in syms.split(',') if s.strip()]
    if not isinstance(syms, list):
        syms = []
    dl['symbols'] = syms

    dl.setdefault('timeframes', [])
    dl.setdefault('start_date', "")
    dl.setdefault('end_date', "")

    return cfg

def save_config(cfg, config_path=CONFIG_PATH):
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)


def save_current_config():
    config = load_config()
    download_config = config.setdefault('download', {})

    # Save symbols
    download_config['symbols'] = [
        symbol.strip() for symbol in symbol_listbox.get(0, tk.END) if symbol.strip()
    ]

    # Save timeframes
    selected_indices = timeframe_listbox.curselection()
    download_config['timeframes'] = [timeframes[i] for i in selected_indices]

    # Save start date/time
    start_date = start_date_entry.get_date()
    start_hour = start_time_spinbox.get().zfill(2)
    start_minute = start_minute_spinbox.get().zfill(2)
    download_config['start_date'] = f"{start_date} {start_hour}:{start_minute}:00"

    # ----------------- CHANGED: end date saving respects "No end date" checkbox -----------------
    if no_end_var.get():
        download_config['end_date'] = ""
    else:
        end_date = end_date_entry.get_date()
        end_hour = end_time_spinbox.get().zfill(2)
        end_minute = end_minute_spinbox.get().zfill(2)
        download_config['end_date'] = f"{end_date} {end_hour}:{end_minute}:00"
    # --------------------------------------------------------------------------------------------

    save_config(config, 'config.json')
    messagebox.showinfo("Config Saved", "Configuration has been saved successfully.")

def start_download():
    save_current_config()
    threading.Thread(target=run_download).start()

import subprocess, sys

def run_download():
    try:
        logging.info("Starting data download...")
        subprocess.run([sys.executable, os.path.join("download", "download.py")], check=True)
        logging.info("Data download completed.")
    except Exception as e:
        logging.error(f"Error during download: {e}")



def add_to_listbox(listbox, entry_widget):
    item = entry_widget.get().strip()
    if item:
        listbox.insert(tk.END, item)
        entry_widget.delete(0, tk.END)

def remove_from_listbox(listbox):
    try:
        index = listbox.curselection()[0]
        listbox.delete(index)
    except IndexError:
        messagebox.showwarning("No Selection", "Please select an item to remove.")

# ----------------- ADDED: helper to enable/disable end-date controls -----------------
def toggle_end_date_controls():
    disabled = no_end_var.get()
    state = 'disabled' if disabled else 'normal'
    end_date_entry.configure(state=state)
    end_time_spinbox.configure(state=state)
    end_minute_spinbox.configure(state=state)
# -------------------------------------------------------------------------------------

# Create the GUI as before, but no logging Text box needed
root = tk.Tk()
root.title("Binance Data Downloader")
root.geometry("800x600")
root.minsize(800, 600)

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

scrollable_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

config = load_config()
download_config = config.get('download', {})

# Symbol Section
symbol_frame = ttk.LabelFrame(scrollable_frame, text="Symbols", padding=(10, 5))
symbol_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

symbol_entry = ttk.Entry(symbol_frame, width=20)
symbol_entry.grid(row=0, column=0, padx=5, pady=5)

symbol_add_button = ttk.Button(symbol_frame, text="Add Symbol",
                               command=lambda: add_to_listbox(symbol_listbox, symbol_entry))
symbol_add_button.grid(row=0, column=1, padx=5, pady=5)

symbol_remove_button = ttk.Button(symbol_frame, text="Remove Symbol",
                                  command=lambda: remove_from_listbox(symbol_listbox))
symbol_remove_button.grid(row=0, column=2, padx=5, pady=5)

symbol_listbox = tk.Listbox(symbol_frame, height=10, width=30, selectmode=tk.SINGLE)
symbol_listbox.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

for symbol in download_config.get('symbols', []):
    symbol_listbox.insert(tk.END, symbol)

# Timeframe Section
timeframe_frame = ttk.LabelFrame(scrollable_frame, text="Timeframes", padding=(10, 5))
timeframe_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d"]
timeframe_listbox = tk.Listbox(timeframe_frame, height=10, width=15, selectmode=tk.MULTIPLE, exportselection=False)
timeframe_listbox.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

for timeframe in timeframes:
    timeframe_listbox.insert(tk.END, timeframe)

scrollbar_timeframe = ttk.Scrollbar(timeframe_frame, orient=tk.VERTICAL, command=timeframe_listbox.yview)
scrollbar_timeframe.grid(row=0, column=1, sticky="ns")
timeframe_listbox.config(yscrollcommand=scrollbar_timeframe.set)

existing_timeframes = download_config.get('timeframes', [])
for i, timeframe in enumerate(timeframes):
    if timeframe in existing_timeframes:
        timeframe_listbox.selection_set(i)

# Date Range Section
date_frame = ttk.LabelFrame(scrollable_frame, text="Date Range (UTC)", padding=(10, 5))
date_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

start_date_label = ttk.Label(date_frame, text="Start Date:")
start_date_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

start_date_entry = DateEntry(date_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
start_date_str = download_config.get('start_date', '2000-01-01 00:00:00')
try:
    start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
except ValueError:
    try:
        start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    except ValueError:
        start_date_obj = datetime(2000, 1, 1).date()
start_date_entry.set_date(start_date_obj)
start_date_entry.grid(row=0, column=1, padx=5, pady=5)

start_time_label = ttk.Label(date_frame, text="Start Time:")
start_time_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")

start_time_spinbox = ttk.Spinbox(date_frame, from_=0, to=23, increment=1, width=5, format="%02.0f")
start_time_spinbox.delete(0, tk.END)
start_time_spinbox.insert(
    0,
    download_config.get('start_date', '').split()[-1].split(":")[0] if ":" in download_config.get('start_date', '') else "00"
)
start_time_spinbox.grid(row=0, column=3, padx=5, pady=5)

start_minute_spinbox = ttk.Spinbox(date_frame, from_=0, to=59, increment=1, width=5, format="%02.0f")
start_minute_spinbox.delete(0, tk.END)
start_minute_spinbox.insert(
    0,
    download_config.get('start_date', '').split()[-1].split(":")[1] if ":" in download_config.get('start_date', '') else "00"
)
start_minute_spinbox.grid(row=0, column=4, padx=5, pady=5)

end_date_label = ttk.Label(date_frame, text="End Date:")
end_date_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

end_date_entry = DateEntry(date_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
end_date_str = download_config.get('end_date', '')

if end_date_str:
    try:
        end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date()
    except ValueError:
        try:
            end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        except ValueError:
            end_date_obj = None
else:
    end_date_obj = None

if end_date_obj:
    end_date_entry.set_date(end_date_obj)
else:
    # ----------------- CHANGED: never set_date("") -----------------
    end_date_entry.set_date(datetime.utcnow().date())
    # ---------------------------------------------------------------

end_date_entry.grid(row=1, column=1, padx=5, pady=5)

end_time_label = ttk.Label(date_frame, text="End Time:")
end_time_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")

end_time_spinbox = ttk.Spinbox(date_frame, from_=0, to=23, increment=1, width=5, format="%02.0f")
end_time_spinbox.delete(0, tk.END)
end_time_spinbox.insert(
    0,
    download_config.get('end_date', '').split()[-1].split(":")[0] if ":" in download_config.get('end_date', '') else "23"
)
end_time_spinbox.grid(row=1, column=3, padx=5, pady=5)

end_minute_spinbox = ttk.Spinbox(date_frame, from_=0, to=59, increment=1, width=5, format="%02.0f")
end_minute_spinbox.delete(0, tk.END)
end_minute_spinbox.insert(
    0,
    download_config.get('end_date', '').split()[-1].split(":")[1] if ":" in download_config.get('end_date', '') else "59"
)
end_minute_spinbox.grid(row=1, column=4, padx=5, pady=5)

# ----------------- ADDED: checkbox to control optional end date -----------------
no_end_var = tk.BooleanVar(value=not bool(end_date_str.strip()))
no_end_check = ttk.Checkbutton(
    date_frame,
    text="No end date (download up to latest)",
    variable=no_end_var,
    command=toggle_end_date_controls
)
no_end_check.grid(row=2, column=0, columnspan=5, padx=5, pady=5, sticky="w")
# Initialize disabled/enabled state based on checkbox
toggle_end_date_controls()
# -------------------------------------------------------------------------------

# Remove the logger_frame and any references to it
# No custom logger_text needed

buttons_frame = ttk.Frame(scrollable_frame)
buttons_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

save_button = ttk.Button(buttons_frame, text="Save Config", command=save_current_config)
save_button.grid(row=0, column=0, padx=10, pady=5)

download_button = ttk.Button(buttons_frame, text="Start Download", command=start_download)
download_button.grid(row=0, column=1, padx=10, pady=5)

scrollable_frame.columnconfigure(0, weight=1)
# Adjust row configs as needed, we removed the logger row
scrollable_frame.rowconfigure(3, weight=0)

root.mainloop()
