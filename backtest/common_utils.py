import json
import winsound

def load_config(file_path: str):
    """
    Load JSON configuration from a file.
    :param file_path: Path to the configuration file
    :return: Dictionary with configuration data
    """
    with open(file_path, 'r') as file:
        return json.load(file)
    
def beep():
    # Change frequency (1000 Hz) and duration (500 ms) as needed
    winsound.Beep(1000, 500)
