import json
from pathlib import Path
import os
from load_profiles import LoadProfile

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

def create_lines():
    data = open_database()
    substations = data['substations']
    lines = []
    
    for substation in substations:
        sub = Substation.get_substation_by_name(substation['properties']['name'])
        connected_AMIs = sub.get_connected_AMIs()
        for AMI in connected_AMIs:
            lines.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [
                        [substation['geometry']['coordinates'][0], substation['geometry']['coordinates'][1]],
                        [AMI['geometry']['coordinates'][0], AMI['geometry']['coordinates'][1]]
                    ]
                },
                'properties': {
                    'substation': substation['properties']['name'],
                    'AMI': AMI['properties']['AMI_id'],
                    "stroke" : "#939393",
                    "gap-size" : 2 
                }
            })
    
    data['lines'] = lines
    save_database(data)
    return


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
                'loads': self.loads,
                'title': self.name
            }
        }

    def get_connected_AMIs(self):
        data = open_database()
        AMIs = data['AMIs']
        connected_AMIs = []

        for AMI in AMIs:
            if AMI['properties']['substation'] == self.name:
                connected_AMIs.append(AMI)
        return connected_AMIs
        
    
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


class AMI:
    def __init__(self, AMI_id, location, customer_type, substation):
        self.AMI_id = AMI_id
        self.location = {'lon': location[0], 'lat': location[1]}  # Swap the order of coordinates
        self.customer_type = customer_type
        self.power_rating = 7.2
        self.substation = substation
        #self.num_feeders = num_feeders
        
        if self.customer_type == 'Household':
            self.load = LoadProfile('Household', 7.2, 'Household').scaled_profile
        if self.customer_type == 'Industrial':
            self.load = LoadProfile('Industrial', 35, 'Industrial').scaled_profile
        if self.customer_type == 'Commercial': 
            self.load = LoadProfile('Commercial', 21, 'Commercial').scaled_profile

    def save_AMI(self):
        data = open_database()
        AMIs = data['AMIs']

        # Find the AMI if it exists and update it
        for i, AMI in enumerate(AMIs):
            if AMI['properties']['AMI_id'] == self.AMI_id:
                AMIs[i] = self.to_geojson()
                break
        else:
            # AMI not found, add it
            AMIs.append(self.to_geojson())

        save_database(data)

    
    def to_geojson(self):
        if self.customer_type == 'Household':
            color = '#ffff00'
        elif self.customer_type == 'Industrial':
            color = '#ff0000'
        elif self.customer_type == 'Commercial':
            color = '#0000ff'
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [self.location['lat'], self.location['lon']]
            },
            'properties': {
                'AMI_id': self.AMI_id,
                'customer_type': self.customer_type,
                'marker-color': color,
                'power_rating': self.power_rating,
                'load': self.load,
                'substation': self.substation,
                'title': self.AMI_id
            }
        }