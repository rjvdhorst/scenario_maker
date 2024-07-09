from pathlib import Path
import db_helpers as db
import json
from viktor.errors import InputViolation, UserError
import csv

class LoadProfile:
    def __init__(self, name, peak_load, base_profile):
        self.name = name
        self.peak_load = peak_load
        self.base_profile = base_profile
        self.normal_profile = self.profile_dict()
        self.scaled_profile = self.scale_profile()
 
    def profile_dict(self):
        base_profiles_folder = Path(__file__).parent / 'base_profiles'
        filename = self.base_profile + '.json'
        json_file = base_profiles_folder / filename
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def scale_profile(self):
        normal_profile = self.normal_profile
        scaled_profile = {'time_array': []}
        for timestep in normal_profile['time_array']:
            scaled_value = {'value': self.peak_load * timestep['value'], 'time': timestep['time']}
            scaled_profile['time_array'].append(scaled_value)
        return scaled_profile

    def save_profile(self):
        data = db.open_database()
        profiles = data['profiles']
        if any(profile['name'] == self.name for profile in profiles):
            raise UserError("Name '" + self.name + "' is already taken")
        profile = {
            'name': self.name,
            'peak_load': self.peak_load,
            'base_profile': self.base_profile
        }
        data['profiles'].append(profile)
        db.save_database(data)

    @staticmethod
    def all_customer_profiles():
        data = db.open_database()
        profiles = data['profiles']
        return profiles
    
    @staticmethod
    def list_names():
        base_profiles_folder = Path(__file__).parent / 'base_profiles'
        json_files = base_profiles_folder.glob('*.json')
        json_file_names = [file.stem for file in json_files]
        return json_file_names

    @staticmethod
    def find_load_profile(name):
        data = db.open_database()
        profiles = data['profiles']
        for profile in profiles:
            if profile['name'] == name:
                return LoadProfile(profile['name'], profile['peak_load'], profile['base_profile'])
        return None
    
class BaseProfile:
    def __init__(self, profile_name):
        self.profile_name = profile_name
        self.data = self.load_profile_data()
    
    def load_profile_data(self):
        base_profiles_folder = Path(__file__).parent / 'base_profiles'
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
        base_profiles_folder = Path(__file__).parent / 'base_profiles'
        file_name = profile_name + '.json'
        json_file = base_profiles_folder / file_name

        json_payload = {
            'name': profile_name,
            'time_array': time_array
        }

        with open(json_file, 'w') as f:
            json.dump(json_payload, f)
