import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

class Database:
    _instance: Optional['Database'] = None
    _lock: threading.Lock = threading.Lock()
    _data_cache: Optional[Dict[str, Any]] = None
    _models_cache: Optional[Dict[str, Any]] = None

    def __new__(cls) -> 'Database':
        """Ensure only one instance of the class is created (Singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Database, cls).__new__(cls)
                    cls._data_cache = None  # Lazy-loaded cache for data.json
                    cls._models_cache = None  # Lazy-loaded cache for models.json
        return cls._instance

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Helper method to load a JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Helper method to save data to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def open_database(self) -> Dict[str, Any]:
        """Lazy load and return the data from data.json."""
        if self._data_cache is None:
            json_file = Path(__file__).parent.parent / 'data.json'
            self._data_cache = self._load_json(json_file)
        return self._data_cache

    def save_database(self) -> None:
        """Save the cached data back to data.json."""
        print('Saving db...')
        if self._data_cache is not None:
            json_file = Path(__file__).parent.parent / 'data.json'
            self._save_json(json_file, self._data_cache)

    def open_models(self) -> Dict[str, Any]:
        """Lazy load and return the data from models.json."""
        if self._models_cache is None:
            json_file = Path(__file__).parent.parent / 'models.json'
            self._models_cache = self._load_json(json_file)
        return self._models_cache

    def save_models(self) -> None:
        """Save the cached models back to models.json."""
        if self._models_cache is not None:
            json_file = Path(__file__).parent.parent / 'models.json'
            self._save_json(json_file, self._models_cache)

    def add_model(self, name: str, target: str, features: List[str], coef: List[float], y_test: List[float], predictions: List[float], MSE: float, MAE: float) -> None:
        """Add a new model to the models.json file and save."""
        models = self.open_models().get('models', [])
        
        # Create the new model dictionary
        new_model: Dict[str, Any] = {
            'model_name': name,
            'target': target,
            'features': features, 
            'coefficients': coef,
            'y_test': y_test,
            'predictions': predictions,
            'MSE': MSE,
            'MAE': MAE
        }

        # Append the new model and save
        models.append(new_model)
        self._models_cache['models'] = models
        self.save_models()

    def clear_cache(self) -> None:
        """Clear both data and models caches."""
        self._data_cache = None
        self._models_cache = None
