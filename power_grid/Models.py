import json
from pathlib import Path
from .Database import Database

def add_model(name, target, features, coef, y_test, predictions, MSE, MAE):
    """Adds a new model to the models database."""
    db = Database()
    data = db.open_models()
    
    models = data['models']
    new_model = {
        'model_name': name,
        'target': target,
        'features': features, 
        'coefficients': coef,
        'y_test': y_test,
        'predictions': predictions,
        'MSE': MSE,
        'MAE': MAE
    }
    models.append(new_model)

    db.save_models(models)
