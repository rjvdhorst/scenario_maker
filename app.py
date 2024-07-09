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
    MultiSelectField, FileField, ActionButton
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

    step_2 = Step("Manage Load Profiles", description="Manage load profiles for different customer groups", views=["get_plotly_view_1"])

    step_2.section_1 = Section("Create Customer Group", description="Customize the specified load profiles. ")
    step_2.section_1.intro = Text("""Create load profiles for different customer types. Specify the name, peak load, and base profile. The base profile is a predefined profile that can be scaled to the peak load.""")

    step_2.section_1.dynamic_array_1 = DynamicArray("")
    step_2.section_1.dynamic_array_1.profile_name = TextField("Name", flex=33)
    step_2.section_1.dynamic_array_1.peak_load = NumberField("Peak Load", suffix='KW', flex=33)
    step_2.section_1.dynamic_array_1.base_profile = OptionField("Base Profile", options=list_base_profiles(), flex=33)
    step_2.section_1.normalize_button = ActionButton('Add to database', flex=100, method='add_load_profile')
    
    step_2.section_2 = Section("Investigate Base Profiles")
    step_2.section_2.introtext = Text("Select a base profile to see the normalized load profile. This profile will be multilpied by the peak load to get the final load profile.")
    step_2.section_2.select_load_profile = OptionField('Select Profile', options=list_base_profiles(), default='Household', flex=100)
    
    step_3 = Step("Develop Energy Landscape", views=['get_map_view_1'])
    step_3.section_1 = Section("Assign Load Profile to Substation/Transformer")
    step_3.section_1.text_1 = Text("""By assigning a customer group and the amount of customers in that group to a substation or transformer, an aggregated load profile for the specific transformer is developed.""")
    step_3.section_1.dynamic_array_1 = DynamicArray("Connections")
    step_3.section_1.dynamic_array_1.substation_name = OptionField("Substation", options=list_substations(), flex=50)
    step_3.section_1.dynamic_array_1.customer_type = OptionField("Customer Group", options=list_customer_profiles(), flex=50)
    step_3.section_1.dynamic_array_1.num_connections = IntegerField("Number of Customers", description="Define how many customers of this type are connected.", flex=50)
    # step_3.section_1.dynamic_array_1.peak_load = NumberField("Peak Load", description="Define the peak load of the connection")
    step_3.section_1.dynamic_array_1.number_field_1 = NumberField("Variance (%)", description="Define the variance in the load profile.", flex=50)
    step_3.section_1.connect_button = ActionButton('Connect', flex=100, method='connect_load')

    step_3.section_2 = Section("Remove Load")
    step_3.section_2.intro = Text('This section allows you to remove previously assigned load from the substation/transformer. Select the loads that need to be removed, and press the button.')
    step_3.section_2.substation_name = OptionField('Substation/Transformer', options=list_substations(), flex=50)
    step_3.section_2.load_name = MultiSelectField('Name', options=list_connected_loads, flex=50)
    step_3.section_2.remove_button = ActionButton('Remove Load', flex=100, method='remove_load')


    step_4 = Step("Future Scenario", description="Create scenarios based on the growrates", views=["get_plotly_view_1"])
    step_4.section_1 = Section("Load Growth Factor", description="Specify the Load Growth Factor per customer group for the substation")
    step_4.section_1.dynamic_array_1 = DynamicArray("Growrate")
    step_4.section_1.dynamic_array_1.customer_type = OptionField("Customer Group", flex=33, description="Select the previously define customer group from the database to apply a load growth factor to.", options=['Household 1', 'Business 1', 'EV - public'])
    step_4.section_1.dynamic_array_1.grow = NumberField("Load Growth Factor", flex=33, description="Growrate of the number of customers")
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
        data = params['step_3']['section_1']['dynamic_array_1']
        for connection in data:
            substation_name = connection['substation_name']
            substation = db.Substation.get_substation_by_name(substation_name)
            
            if substation is None:
                raise UserError(f"Substation {substation_name} not found")
            
            num_connections = connection['num_connections']
            load = ph.LoadProfile.find_load_profile(connection['customer_type'])
            
            # print(load.scaled_profile)
            substation.add_load(load, num_connections)

    def remove_load(self, params, **kwargs):
        substation_name = params['step_3']['section_2']['substation_name']
        load_name = params['step_3']['section_2']['load_name']
        substation = db.Substation.get_substation_by_name(substation_name)
        substation.remove_load(load_name)
        

    @staticmethod
    def list_connected_loads(params, **kwargs):
        substation_name = params['step_3']['section_2']['substation_name']
        substation = db.Substation.get_substation_by_name(substation_name)
        return [load['name'] for load in substation.loads]
    

    @PlotlyView('Base Load Profile', duration_guess=1)
    def get_plotly_view_1(self, params, **kwargs):
        profile_name = params['step_2']['section_2']['select_load_profile']

        if profile_name == None:
            profile_name = 'Household'
        
        profile  = ph.BaseProfile(profile_name).load_profile_data()

        time = [entry['time'] for entry in profile['time_array']]
        values = [entry['value'] for entry in profile['time_array']]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=values, mode='lines', name='Load'))
        
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
    
    
    @MapView('Energy Asset Overview', duration_guess=1)
    def get_map_view_1(self, params, **kwargs):

        data = db.open_database()

        # Show all substations
        features = []
        for substation in data['substations']:
            feature = MapPoint(substation['geometry']['coordinates'][1], substation['geometry']['coordinates'][0], description=substation['properties']['name'], identifier=substation['properties']['name'])
            features.append(feature)

        return MapResult(features)
    
    @DataView('Summary', duration_guess=1)
    def overview_data(self, params, **kwargs):
        data = DataGroup(
            group_a=DataItem('Total Demand', '', subgroup=DataGroup(
                sub_group1=DataItem('Average Power Demand', 11, suffix='MW', subgroup=DataGroup(
                    value_a=DataItem('Households', 5, suffix='MW'),
                    value_b=DataItem('Business', 5, suffix='MW'),
                    value_c=DataItem('EV', 1, suffix='MW'))), 
                sub_group2=DataItem('Max Power Demand', 9, suffix='MW', subgroup=DataGroup(
                    value_a=DataItem('Households', 3, suffix='MW'),
                    value_b=DataItem('Business', 5, suffix='MW'),
                    value_c=DataItem('EV', 1, suffix='MW')),
            ))),
        )
        return DataResult(data)
    

    @PDFView("Report Viewer", duration_guess=1)
    def get_pdf_view(self, params, **kwargs):
        file_path = Path(__file__).parent / 'tuto2.pdf'
        return PDFResult.from_path(file_path)
    
