from viktor.parametrization import (
    ViktorParametrization,
    TextField,
    NumberField,
    Table,
    Section,
    Step,
    Text,
    OptionField,
    IntegerField,
    DynamicArray,
    GeoPointField, 
    GeoPolygonField,
    BooleanField,
    LineBreak,
    MultiSelectField, FileField, ActionButton, SetParamsButton
)
from pathlib import Path
from viktor import ViktorController
from viktor.views import (
    PlotlyView,
    MapView,
    PlotlyResult,
    MapResult,
    DataView, DataResult, DataGroup, DataItem, PDFResult, PDFView, MapPoint
)

from viktor.errors import UserError
from viktor.result import SetParamsResult

import plotly.graph_objects as go
import db_helpers as db
import load_profiles as ph

def list_substations(**kwargs):
    data = db.open_database()
    substations = data['substations']
    if not substations:
        return ['No substations']
    substation_names = [substation['properties']['name'] for substation in substations] # Extract the name from each substation
    return substation_names


def list_base_profiles(**kwargs):
    return ph.LoadProfile.list_names()

def list_customer_profiles(**kwargs):
    customer_dict = ph.LoadProfile.all_customer_profiles()
    return [profile['name'] for profile in customer_dict]

def list_connected_loads(params, **kwargs):
    substation_name = params.step_3.section_2.substation_name
    substation = db.Substation.get_substation_by_name(substation_name)
    if substation is not None:
        return [load['name'] for load in substation.loads]
    else:
        return []

def create_default_content():
    load_profile = ph.LoadProfile.find_load_profile('Industrial')
    default_content = load_profile.profile_dict()
    default_array = default_content['time_array']
    return default_array


class Parametrization(ViktorParametrization):
    step_1 = Step("Manage Substations/Transformers", views=["get_map_view_1"])
    step_1.section_1 = Section("Add Substations/Transformers")
    step_1.section_1.intro = Text('Add a new substation to the database. Specify the name and location of the substation or transformer.')
    step_1.section_1.substation_name = TextField('#### Name', flex = 100)
    # step_1.section_1.substation_power = IntegerField('#### Powerrating', flex=33)
    # step_1.section_1.number_of_feeders = IntegerField('#### Number of feeders', flex=33)
    step_1.section_1.substation_location = GeoPointField('#### Location')
    step_1.section_1.add_button = ActionButton('Add to Database', flex=100, method='add_substation')

    step_1.section_2 = Section("Remove Substations/Transformers", description="Remove")
    step_1.section_2.substation_name = OptionField('Name', options=list_substations(), flex=50)
    step_1.section_2.remove_button = ActionButton('Remove from Database', flex=50, method='remove_substation')

    step_2 = Step("Manage Load Profiles", description="Manage load profiles for different customer groups", views=['plotly_new_load_profile', "get_plotly_view_1"])

    step_2.section_1 = Section("Create Customer Group", description="Customize the specified load profiles. ")
    step_2.section_1.intro = Text("""Create load profiles for different customer types. Specify the name, peak load, and base profile. The base profile is a predefined profile that can be scaled to the peak load.""")

    step_2.section_1.dynamic_array_1 = DynamicArray("")
    step_2.section_1.dynamic_array_1.profile_name = TextField("Name", flex=33)
    step_2.section_1.dynamic_array_1.peak_load = NumberField("Peak Load", suffix='KW', flex=33)
    step_2.section_1.dynamic_array_1.base_profile = OptionField("Base Profile", options=list_base_profiles(), flex=33)
    step_2.section_1.normalize_button = ActionButton('Add to database', flex=100, method='add_load_profile')
    
    step_2.section_2 = Section("Add Base Profiles")
    step_2.section_2.introtext = Text("A base profile can be set by uploading a CSV file with the load profile data. The CSV file should contain two columns: time and load. Give a name to the profile and upload the file.")
    step_2.section_2.profile_name = TextField("##### Name", flex=80)
    step_2.section_2.table = Table("Create a New Load Profile", default=create_default_content())
    step_2.section_2.table.time = TextField('time')
    step_2.section_2.table.value = NumberField('value')
    step_2.section_2.upload_button = ActionButton("Save Profile", flex=60, method='save_base_profile')

    step_2.section_3 = Section("View Base Load Profile")
    step_2.section_3.select_load_profile = OptionField("Select Load Profile", options=list_base_profiles(), default='Household', flex=50)
    
    step_3 = Step("Develop Energy Landscape", views=['get_plotly_view_2','get_map_view_1'])
    step_3.section_1 = Section("Assign Load Profile to Substation/Transformer")
    step_3.section_1.text_1 = Text("""By assigning a customer group and the amount of customers in that group to a substation or transformer, an aggregated load profile for the specific transformer is developed.""")
    step_3.section_1.substation_name = OptionField("Substation", options=list_substations(), flex=50)
    step_3.section_1.dynamic_array_1 = DynamicArray("### Connections \n Add new loads to the selected substation")
    step_3.section_1.dynamic_array_1.customer_type = OptionField("Customer Group", options=list_customer_profiles(), flex=50)
    step_3.section_1.dynamic_array_1.num_connections = IntegerField("Number of Customers", description="Define how many customers of this type are connected.", flex=50)
    step_3.section_1.connect_button = SetParamsButton('Connect', flex=100, method='connect_load')

    step_3.section_2 = Section("Remove Load")
    step_3.section_2.intro = Text('This section allows you to remove previously assigned load from the substation/transformer. Select the loads that need to be removed, and press the button.')
    step_3.section_2.substation_name = OptionField('Substation/Transformer', options=list_substations(), flex=50)
    step_3.section_2.load_name = MultiSelectField('Name', options=list_connected_loads, flex=50)
    step_3.section_2.remove_button = ActionButton('Remove Load', flex=100, method='remove_load')

    step_4 = Step("Load Growth Factor", description="Create scenarios based on the growrates", views=["get_plotly_view_1"])
    step_4.section_1 = Section("Load Growth Factor", description="Specify the Load Growth Factor per customer group for the substation")
    step_4.section_1.select_substation = OptionField("Substation", options=list_substations(), flex=50)
    step_4.section_1.dynamic_array_1 = DynamicArray("Growrate")
    step_4.section_1.dynamic_array_1.customer_type = OptionField("Customer Group", flex=50, description="Select the previously define customer group from the database to apply a load growth factor to.", options=list_customer_profiles())
    step_4.section_1.dynamic_array_1.grow = NumberField("Load Growth Factor", flex=50, description="Growrate of the number of customers")
    #step_4.section_1.dynamic_array_1.grow_type = OptionField("Type", flex=33, description="Growth type - percentage (baseline 100%) or absolute (number of connections)", options=['%', 'Absolute'], default='%')
    

class Controller(ViktorController):
    label = "My Entity Type"
    parametrization = Parametrization

    def add_substation(self, params, **kwargs):
        name = params['step_1']['section_1']['substation_name']
        #power_rating = params['step_1']['section_1']['substation_power']
        #num_feeders = params['step_1']['section_1']['number_of_feeders']
        location = (params['step_1']['section_1']['substation_location'].lat, params['step_1']['section_1']['substation_location'].lon)
        db.Substation(name, location).save_substation()

    def remove_substation(self, params, **kwargs):
        name = params['step_1']['section_2']['substation_name']
        db.Substation.remove_substation(name)

    def add_load_profile(self, params, **kwargs):
        data = params['step_2']['section_1']['dynamic_array_1']
        for profile in data:
            load = ph.LoadProfile(profile['profile_name'], profile['peak_load'], profile['base_profile'])
            load.save_profile()

    def connect_load(self, params, **kwargs):
        substation_name = params['step_3']['section_1']['substation_name']
        data = params['step_3']['section_1']['dynamic_array_1']
        for connection in data:
            substation = db.Substation.get_substation_by_name(substation_name)
            
            if substation is None:
                raise UserError(f"Substation {substation_name} not found")
            
            num_connections = connection['num_connections']
            load = ph.LoadProfile.find_load_profile(connection['customer_type'])
            
            # print(load.scaled_profile)
            substation.add_load(load, num_connections)
            
        result = SetParamsResult({
            "step_3": {
                "section_1": {
                    "dynamic_array_1": None
                }
            }
        })
        print(result)
        return result
        

    def remove_load(self, params, **kwargs):
        substation_name = params['step_3']['section_2']['substation_name']
        load_name = params['step_3']['section_2']['load_name']
        substation = db.Substation.get_substation_by_name(substation_name)
        substation.remove_load(load_name)
        
    def save_base_profile(self, params, **kwargs):
        profile_name = params['step_2']['section_2']['profile_name']

        profile = params['step_2']['section_2']['table']
        time_array = []
        for entry in profile:
            time_array.append({'time': entry['time'], 'value': entry['value']})

        ph.BaseProfile.save_base_profile(time_array, profile_name)

    @staticmethod
    def list_connected_loads(params, **kwargs):
        substation_name = params['step_3']['section_2']['substation_name']
        substation = db.Substation.get_substation_by_name(substation_name)
        return [load['name'] for load in substation.loads]
    

    @PlotlyView('Base Load Profile', duration_guess=1)
    def get_plotly_view_1(self, params, **kwargs):
        profile_name = params['step_2']['section_3']['select_load_profile']

        if profile_name == None:
            profile_name = 'Household'
        
        profile  = ph.BaseProfile(profile_name).load_profile_data()

        time = [entry['time'] for entry in profile['time_array']]
        values = [entry['value'] for entry in profile['time_array']]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=time, y=values, name='Load'))
        
        fig.update_layout(
            title= profile['name'] + ' - Load Profile',
            xaxis_title='Hour of the Day',
            yaxis_title='Normalized Load',
            xaxis=dict(
            tickmode='array',
            tickvals=[time[i] for i in range(0, len(time), 12)],  # Every three hours (12 * 15 minutes = 3 hours)
            ticktext=[time[i] for i in range(0, len(time), 12)],
            range=[0, len(time)-1]
            ),
            yaxis=dict(
            range=[0, 1]
            )
        )

        fig = fig.to_json()
        
        return PlotlyResult(fig)

    @PlotlyView('Edited Load Profile', duration_guess=1)
    def plotly_new_load_profile(self, params, **kwargs):
        profile_name = params['step_2']['section_3']['select_load_profile']

        if profile_name == None:
            profile_name = 'Industrial'
        
        profile  = params['step_2']['section_2']['table']

        time = [entry['time'] for entry in profile]
        values = [entry['value'] for entry in profile]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=time, y=values, name='Load'))
        
        fig.update_layout(
            title= profile_name + ' - Load Profile',
            xaxis_title='Hour of the Day',
            yaxis_title='Normalized Load',
            xaxis=dict(
            tickmode='array',
            tickvals=[time[i] for i in range(0, len(time), 12)],  # Every three hours (12 * 15 minutes = 3 hours)
            ticktext=[time[i] for i in range(0, len(time), 12)],
            range=[0, len(time)-1]
            ),
            yaxis=dict(
            range=[0, 1]
            )
        )

        fig = fig.to_json()
        
        return PlotlyResult(fig)
    
    
    @MapView('Energy Asset Overview', duration_guess=10)
    def get_map_view_1(self, params, **kwargs):

        data = db.open_database()

        # Show all substations
        features = []
        for substation in data['substations']:
            feature = MapPoint(substation['geometry']['coordinates'][1], substation['geometry']['coordinates'][0], description=substation['properties']['name'], identifier=substation['properties']['name'])
            features.append(feature)

        return MapResult(features)
    
    @PlotlyView('Aggregated Load', duration_guess=10)
    def get_plotly_view_2(self, params, **kwargs):
        substation_name = params['step_3']['section_1']['substation_name']

        substation = db.Substation.get_substation_by_name(substation_name)
        if substation is None:
            raise UserError(f"Substation {substation_name} not found")

        detailed_profiles = substation.get_detailed_load_profiles()
        time = sorted(detailed_profiles.keys())

        fig = go.Figure()

        # Collecting all load names
        load_names = set()
        for loads_at_time in detailed_profiles.values():
            load_names.update(loads_at_time.keys())
        
        # Create a trace for each load
        for load_name in load_names:
            values = [detailed_profiles[t].get(load_name, 0) for t in time]
            fig.add_trace(go.Bar(x=time, y=values, name=load_name))

        fig.update_layout(
            title='Aggregated Load Profile',
            xaxis_title='Hour of the Day',
            yaxis_title='Normalized Load',
            barmode='stack',
            xaxis=dict(
                tickmode='array',
                tickvals=[time[i] for i in range(0, len(time), 12)],  # Every three hours (12 * 15 minutes = 3 hours)
                ticktext=[time[i] for i in range(0, len(time), 12)],
                range=[0, len(time)-1]
            )
        )

        fig = fig.to_json()
        return PlotlyResult(fig)




        
