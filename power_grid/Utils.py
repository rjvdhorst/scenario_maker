
from .Database import Database
from .Entities import Substation

def create_lines():
    """Generates power lines connecting substations and connections."""
    data = Database.open_database()
    substations = data['substations']
    lines = []

    for substation in substations:
        sub = Substation.get_substation_by_name(substation['properties']['name'])
        connected_connections = sub.get_connected_connections()

        for connection in connected_connections:
            line = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [
                        [substation['geometry']['coordinates'][0], substation['geometry']['coordinates'][1]],
                        [connection['geometry']['coordinates'][0], connection['geometry']['coordinates'][1]]
                    ]
                },
                'properties': {
                    'substation': substation['properties']['name'],
                    'connection': connection['properties']['connection_id'],
                    "stroke": "#939393",
                    "gap-size": 2 
                }
            }
            lines.append(line)
    
    data['lines'] = lines
    save_database(data)
