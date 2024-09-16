from .Database import Database
from .Models import add_model
from .Entities import Substation, Connection
from .Profiles import LoadProfile, BaseProfile
from .Utils import create_lines

# Initialize the Database singleton instance when the package is imported
db_instance = Database()

# Export only the necessary entities and functions
__all__ = [
    'db_instance',  # Singleton database instance
    'add_model',  # Model functions
    'Substation', 'Connection',  # Entity classes
    'LoadProfile', 'BaseProfile',  # Profile classes
    'create_lines'  # Utility functions
]
