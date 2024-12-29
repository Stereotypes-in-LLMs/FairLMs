import json
def parse_config(config_path):
    """
    Parse given config 
    """
    data = None
    with open(config_path, 'r') as file:
        data = json.load(file)
    return data