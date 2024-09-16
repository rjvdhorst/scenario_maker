from .Database import Database
from viktor.errors import UserError
from pathlib import Path
import json

class LoadProfile:
    """Represents a load profile with methods for scaling and saving profiles."""
    
    def __init__(self, name, peak_load, base_profile):
        self.name = name
        self.peak_load = peak_load
        self.base_profile = base_profile
        self.normal_profile = self.profile_dict()
        self.scaled_profile = self.scale_profile()

    def profile_dict(self):
        """Loads the base profile data from a JSON file."""
        base_profiles_folder = Path(__file__).parent.parent / 'base_profiles'
        filename = self.base_profile + '.json'
        json_file = base_profiles_folder / filename
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def scale_profile(self):
        """Scales the profile based on the peak load."""
        normal_profile = self.normal_profile
        scaled_profile = {'time_array': []}
        for timestep in normal_profile['time_array']:
            scaled_value = {'value': self.peak_load * timestep['value'], 'time': timestep['time']}
            scaled_profile['time_array'].append(scaled_value)
        return scaled_profile

    def save_profile(self):
        """Saves the load profile to the database."""
        db = Database()
        data = db.open_database()  # Use the singleton instance
        profiles = data['profiles']
        if any(profile['name'] == self.name for profile in profiles):
            raise UserError(f"Name '{self.name}' is already taken")
        profile = {
            'name': self.name,
            'peak_load': self.peak_load,
            'base_profile': self.base_profile
        }
        profiles.append(profile)
        db.save_database(data)

    @staticmethod
    def all_customer_profiles():
        """Returns a list of all customer profiles."""
        db = Database() # Use the singleton instance
        data = db.open_database()
        profiles = data['profiles']
        return profiles

    @staticmethod
    def list_names():
        """Lists the names of all base profile JSON files."""
        base_profiles_folder = Path(__file__).parent.parent / 'base_profiles'
        json_files = base_profiles_folder.glob('*.json')
        json_file_names = [file.stem for file in json_files]
        return json_file_names

    @staticmethod
    def find_load_profile(name):
        """Finds and returns a LoadProfile by name."""
        db = Database()  # Use the singleton instance
        data = db.open_database()
        profiles = data['profiles']
        for profile in profiles:
            if profile['name'] == name:
                return LoadProfile(profile['name'], profile['peak_load'], profile['base_profile'])
        return None


class BaseProfile:
    """Represents a normalized profile used for generating load profiles."""

    def __init__(self, profile_name):
        self.profile_name = profile_name
        self.data = self.load_profile_data()

    def load_profile_data(self):
        """Loads base profile data from a JSON file."""
        base_profiles_folder = Path(__file__).parent.parent / 'base_profiles'
        file_name = self.profile_name + '.json'
        json_file = base_profiles_folder / file_name
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file '{file_name}' not found.")
            data = None
        return data

    @staticmethod
    def save_base_profile(time_array, profile_name):
        """Saves a new base profile to a JSON file."""
        base_profiles_folder = Path(__file__).parent.parent / 'base_profiles'
        file_name = profile_name + '.json'
        json_file = base_profiles_folder / file_name

        json_payload = {
            'name': profile_name,
            'time_array': time_array
        }

        with open(json_file, 'w') as f:
            json.dump(json_payload, f)