import json
from pathlib import Path
import os

def open_database():
    json_file = Path(__file__).parent / 'data.json'

    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_database(data):
    json_file = Path(__file__).parent / 'data.json'

    with open(json_file, 'w') as f:
        json.dump(data, f)

def open_models():
    json_file = Path(__file__).parent / 'models.json'

    with open(json_file, 'r') as f:
        data = json.load(f)
        
    return data

def add_model(name, features, coef, y_test, predictions, MSE, MAE):
    json_file = Path(__file__).parent / 'models.json'
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    models = data['models']
    models.append({
        'model_name' : name,
        'features' : features, 
        'coefficients' : coef,
        'y_test' : y_test,
        'predictions' : predictions,
        'MSE' : MSE,
        'MAE' : MAE
        })
    
    with open(json_file, 'w') as f:
        json.dump(data, f)

class Substation:
    def __init__(self, name, location):
        self.name = name
        self.location = {'lon': location[0], 'lat': location[1]}  # Swap the order of coordinates
        #self.power_rating = power_rating
        #self.num_feeders = num_feeders
        self.loads = []

    def save_substation(self):
        data = open_database()
        substations = data['substations']

        # Find the substation if it exists and update it
        for i, substation in enumerate(substations):
            if substation['properties']['name'] == self.name:
                substations[i] = self.to_geojson()
                break
        else:
            # Substation not found, add it
            substations.append(self.to_geojson())

        save_database(data)

    @staticmethod
    def remove_substation(name):
        data = open_database()
        substations = data['substations']

        # Remove the substation if it exists
        substations = [sub for sub in substations if sub['properties']['name'] != name]

        data['substations'] = substations
        save_database(data)

    def add_load(self, load, num_connections):
        temp_load = {
            'name': load.name,
            'num_connections': num_connections,
            'profile': load.scaled_profile
        }
        self.loads.append(temp_load)
        self.save_substation()

    def remove_load(self, load_names):
        for load_name in load_names:
            for load in self.loads:
                if load['name'] == load_name:
                    self.loads.remove(load)
                    self.save_substation()
                    break
        
    def get_total_load(self):
        total_load_profile = {}

        if not self.loads:
            return total_load_profile

        for load in self.loads:
            print(load['name'])
            load_profile = load['profile']['time_array']
            for timestep in load_profile:
                if timestep['time'] not in total_load_profile:
                    total_load_profile[timestep['time']] = 0
                total_load_profile[timestep['time']] += timestep['value']*load['num_connections']
        
        total_load_profile['name'] = self.name
        
        return total_load_profile
    
    def get_detailed_load_profiles(self):
        """Returns detailed load profiles for each load."""
        detailed_profiles = {}

        for load in self.loads:
            load_profile = load['profile']['time_array']
            for timestep in load_profile:
                time = timestep['time']
                if time not in detailed_profiles:
                    detailed_profiles[time] = {}
                detailed_profiles[time][load['name']] = timestep['value'] * load['num_connections']

        return detailed_profiles

    def to_geojson(self):
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [self.location['lat'], self.location['lon']]
            },
            'properties': {
                'name': self.name,
                #'power_rating': self.power_rating,
                #'num_feeders': self.num_feeders,
                'loads': self.loads
            }
        }
        
    
    @classmethod
    def get_substation_by_name(cls, name):
        data = open_database()
        substations = data['substations']

        for substation in substations:
            if substation['properties']['name'] == name:
                location = [substation['geometry']['coordinates'][1], substation['geometry']['coordinates'][0]]
                #power_rating = substation['properties']['power_rating']
                #num_feeders = substation['properties']['num_feeders']
                loads = substation['properties']['loads']

                substation_obj = cls(name, location)
                substation_obj.loads = loads
                return substation_obj

        return None 


