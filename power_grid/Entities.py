from .Database import Database
from .Profiles import LoadProfile

class Substation:
    """Represents a substation with its operations."""

    def __init__(self, name, location):
        self.name = name
        self.location = {'lon': location[0], 'lat': location[1]}

    def to_geojson(self):
        """Converts substation data to GeoJSON format."""
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [self.location['lat'], self.location['lon']]
            },
            'properties': {
                'name': self.name,
                'title': self.name
            }
        }

    def get_connected_connections(self):
        """Returns a list of Connections connected to this substation."""
        db = Database()
        connections = []
        data = db.open_database()  # Access cached data
        for connection in data['connections']:
            if connection['properties']['substation'] == self.name:
                location = [connection['geometry']['coordinates'][1], connection['geometry']['coordinates'][0]]
                connection_obj = Connection(connection['properties']['connection_id'], location, connection['properties']['customer_type'], self)
                connections.append(connection_obj)
        return connections

    @classmethod
    def get_substation_by_name(cls, name):
        """Retrieves a substation by name from the database."""
        db = Database()
        data = db.open_database()  # Access cached data
        for substation in data['substations']:
            if substation['properties']['name'] == name:
                location = [substation['geometry']['coordinates'][1], substation['geometry']['coordinates'][0]]
                return cls(name, location)
        return None

    @classmethod
    def save_substation(cls, substation):
        """Saves or updates a substation in the database."""
        db = Database()
        data = db.open_database()  # Access cached data
        substations = data['substations']

        # Update or add the substation
        for i, existing_substation in enumerate(substations):
            if existing_substation['properties']['name'] == substation.name:
                substations[i] = substation.to_geojson()
                break
        else:
            substations.append(substation.to_geojson())

        db.save_database()  # Persist changes

    @classmethod
    def remove_substation(cls, name):
        """Removes a substation from the database."""
        db = Database()
        data = db.open_database()  # Access cached data
        substations = [sub for sub in data['substations'] if sub['properties']['name'] != name]
        data['substations'] = substations
        db.save_database()  # Persist changes


class Connection:
    """Represents a Connection with its properties and load profile."""

    def __init__(self, connection_id, location, customer_type, substation):
        self.connection_id = connection_id
        self.location = {'lon': location[0], 'lat': location[1]}
        self.customer_type = customer_type
        self.substation = substation

        # Set power rating and load profile based on customer type
        self.power_rating = {'Household': 7.2, 'Industrial': 35, 'Commercial': 21}[customer_type]
        self.load = LoadProfile(customer_type, self.power_rating, customer_type).scaled_profile

    def to_geojson(self):
        """Converts connection data to GeoJSON format."""
        color = {'Household': '#ffff00', 'Industrial': '#ff0000', 'Commercial': '#0000ff'}[self.customer_type]
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [self.location['lat'], self.location['lon']]
            },
            'properties': {
                'connection_id': self.connection_id,
                'customer_type': self.customer_type,
                'marker-color': color,
                'power_rating': self.power_rating,
                'load': self.load,
                'substation': self.substation,
                'title': self.connection_id,
                'description': f"Type: {self.customer_type}<br>Power Rating: {self.power_rating} kVA"
            }
        }

    @classmethod
    def save_connection(cls, connection):
        """Saves or updates a connection in the database."""
        db = Database()
        data = db.open_database()  # Access cached data
        connections = data['connections']

        # Update or add the connection
        for i, existing_connection in enumerate(connections):
            if existing_connection['properties']['connection_id'] == connection.connection_id:
                connections[i] = connection.to_geojson()
                break
        else:
            connections.append(connection.to_geojson())

        db.save_database()  # Persist changes

    @classmethod
    def get_connection_by_id(cls, connection_id):
        """Retrieves a connection by ID from the database."""
        db = Database()
        data = db.open_database()  # Access cached data
        for connection in data['connections']:
            if connection['properties']['connection_id'] == connection_id:
                location = [connection['geometry']['coordinates'][1], connection['geometry']['coordinates'][0]]
                substation = Substation.get_substation_by_name(connection['properties']['substation'])
                return cls(connection_id, location, connection['properties']['customer_type'], substation)
        return None

    @classmethod
    def remove_connection(cls, connection_id):
        """Removes a connection from the database."""
        db = Database()
        data = db.open_database()  # Access cached data
        connections = [conn for conn in data['connections'] if conn['properties']['connection_id'] != connection_id]
        data['connections'] = connections
        db.save_database()  # Persist changes
