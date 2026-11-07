import json
def parse_config(config_path):
    """
    Parse given config 
    """
    data = None
    with open(config_path, 'r') as file:
        data = json.load(file)
    return data

def model_revision(model_path, models_file=None):
    """
    Pinned HF commit for the model (configs/models.json), or None for local paths / unlisted models
    """
    from pathlib import Path
    path = Path(models_file or Path(__file__).resolve().parents[1] / "configs" / "models.json")
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(model_path, {}).get("revision")
