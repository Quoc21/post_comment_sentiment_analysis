import os

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ./src
DATA_DIR = os.path.join(BASEDIR, '..', 'data') # ./data


CONFIG = {
    'fb_state_path': os.path.join(BASEDIR, 'crawler', 'resources', 'state.json'),
    'raw_data_path': os.path.join(DATA_DIR, 'raw', 'data.json'),
    'preprocessed_data_path': os.path.join(DATA_DIR, 'processed', 'preprocessed_data.json'),
    'abbreviation_path': os.path.join(BASEDIR, 'data_processing', 'resources', 'abbreviation.json'),
    'emoji_path': os.path.join(BASEDIR, 'data_processing', 'resources', 'emoji_vi.json'),
}