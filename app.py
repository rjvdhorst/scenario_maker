from viktor.parametrization import (
    ViktorParametrization,
    TextField,
    NumberField,
    FileField,
    Table,
    Section,
    Step,
    Tab,
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
from io import BytesIO
from viktor import ViktorController, Color

from viktor.views import (
    PlotlyView,
    MapView,
    PlotlyResult,
    MapResult,
    DataView, DataResult, DataGroup, DataItem, PDFResult, PDFView, MapPoint, TableView, TableResult, GeoJSONResult, GeoJSONView
)

from viktor.errors import UserError
from viktor.result import SetParamsResult

import plotly.graph_objects as go
import db_helpers as db
import load_profiles as ph

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing
import numpy
import pickle
from plotly.subplots import make_subplots
from statistics import mean


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

def list_column_names(params, **kwargs):
    if params["step_0"]["tab_train"]["file"]:
        upload_file = params["step_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        return list(df.columns)
    else:
        return []

def create_default_content():
    load_profile = ph.LoadProfile.find_load_profile('Industrial')
    default_content = load_profile.profile_dict()
    default_array = default_content['time_array']
    return default_array

class Parametrization(ViktorParametrization):
    step_0 = Step("Load Growth Factor Regression", views = ["get_table_view", "get_data_view", "get_predict_view", "get_forecast_view"], width=30)
    # TODO: Tab 0 Upload File
    step_0.tab_train = Tab("[I] Train Model")
    step_0.tab_train.file = FileField("Upload File", file_types=[".csv"])
    step_0.tab_train.features = MultiSelectField('Select Features', options=list_column_names, flex=50)
    step_0.tab_train.target = MultiSelectField('Select Target', options=list_column_names, flex=50)
    step_0.tab_train.testset = NumberField("Test Sample Size", min=0.2, max=0.5, step =0.1, variant='slider')
    step_0.tab_train.model_name = TextField("Model Name")
    step_0.tab_train.train = ActionButton("Train Model", method = 'train_model')

    step_0.tab_evaluate = Tab("[II] Evaluate Model")
    step_0.tab_evaluate.model_name = TextField('Model Name')
    
    step_0.tab_forecast = Tab("[III] Forecast")
    step_0.tab_forecast.model_name = TextField('Model Name')

    step_1 = Step("Manage Substations/Transformers", views=["get_map_view_1"])
    step_1.section_1 = Section("Add Substations/Transformers")
    step_1.section_1.intro = Text('Add a new substation to the database. Specify the name and location of the substation or transformer.')
    step_1.section_1.substation_name = TextField('#### Name', flex = 100)
    # step_1.section_1.substation_power = IntegerField('#### Powerrating', flex=33)
    # step_1.section_1.number_of_feeders = IntegerField('#### Number of feeders', flex=33)
    step_1.section_1.substation_location = GeoPointField('#### Location')
    step_1.section_1.add_button = ActionButton('Add to Database', flex=100, method='add_substation')

    step_1.section_2 = Section("Add AMI", description="Add an AMI to the database")
    step_1.section_2.ami_id = TextField('AMI ID', flex=50)
    step_1.section_2.ami_location = GeoPointField('Location')
    step_1.section_2.customer_type = OptionField('Customer Type', options=['Household', 'Industrial', 'Commercial'], flex=50)
    step_1.section_2.substation = OptionField('Substation', options=list_substations(), flex=50)
    step_1.section_2.add_button = ActionButton('Add to Database', flex=100, method='add_ami')

    step_1.section_3 = Section("Remove Substations/Transformers", description="Remove")
    step_1.section_3.substation_name = OptionField('Name', options=list_substations(), flex=50)
    step_1.section_3.remove_button = ActionButton('Remove from Database', flex=50, method='remove_substation')

    step_2 = Step("Manage Load Profiles", description="Manage load profiles for different customer groups", views=['plotly_new_load_profile', "get_plotly_view_1"])

    step_2.section_1 = Section("Create Customer Group", description="Customize the specified load profiles. ")
    step_2.section_1.intro = Text("""Create load profiles for different customer types. Specify the name, peak load, and base profile. The base profile is a predefined profile that can be scaled to the peak load.""")

    step_2.section_1.dynamic_array_1 = DynamicArray("")
    step_2.section_1.dynamic_array_1.profile_name = TextField("Name", flex=33)
    step_2.section_1.dynamic_array_1.peak_load = NumberField("Peak Load", suffix='KW', flex=33)
    step_2.section_1.dynamic_array_1.base_profile = OptionField("Base Load Profile", options=list_base_profiles(), flex=33)
    step_2.section_1.normalize_button = ActionButton('Add to database', flex=100, method='add_load_profile')
    
    step_2.section_2 = Section("Add Base Load Profiles")
    step_2.section_2.introtext = Text("A Base Load Profile can be configured by filling the table below. Note that the profile is a normalized profile. A value of 1 is equal to the peak load that will be assigned to the profile in a next step.")
    step_2.section_2.profile_name = TextField("##### Name", flex=80)
    step_2.section_2.table = Table("Create a New Base Load Profile", default=create_default_content())
    step_2.section_2.table.time = TextField('time')
    step_2.section_2.table.value = NumberField('value')
    step_2.section_2.upload_button = ActionButton("Save Base Load Profile", flex=60, method='save_base_profile')

    step_2.section_3 = Section("View Base Load Profile")
    step_2.section_3.select_load_profile = OptionField("Select Base Load Profile", options=list_base_profiles(), default='Household', flex=50)
    
    step_3 = Step("Develop Energy Landscape", views=['get_map_view_1', 'get_plotly_view_2'])
    step_3.section_1 = Section("Load growth factor", description="Assign a load growth factor to each customer group for the selected substation.")
    step_3.section_1.text_1 = Text("""To carefully predict the load on the substation, a different load growth factor can be assigned to each customer group.""")
    step_3.section_1.substation_name = OptionField("Substation", options=list_substations(), flex=50)
    step_3.section_1.table = Table("### Load Growth Factor \n Add load to the selected substation", default=[])
    step_3.section_1.table.customer_type = OptionField("Customer Group", options=['Household', 'Industrial', 'Commercial'])
    step_3.section_1.table.lgf = IntegerField("Load Growth Factor (percentage)", description="Define how many customers of this type are connected.")

    # step_3.section_1.connect_button = SetParamsButton('Connect', flex=100, method='connect_load')

    step_3.section_2 = Section("Rooftop Solar", description="Assign a peak power production load to substation, based on the number of solar panels installed.")
    step_3.section_2.intro = Text('This section allows you to add and simulate the impact of rooftop solar panels on the substation.')
    step_3.section_2.remove_button = NumberField('Load', suffix='KW', flex=50)

    step_3.section_3 = Section("EV charging landscape", description="Add EV charging stations to the substation. Choose between the different types of charging stations and their corresponding chargin behaviour.")
    step_3.section_3.intro = Text('This section allows you to add and simulate the impact of EV charging stations on the substation.')
    step_3.section_3.array = DynamicArray("EV Charging Stations")
    step_3.section_3.array.type = OptionField("#### Type", options=['Slow (7 KW)', 'Fast (22 KW)', 'Ultra Fast (70 KW)'], flex=50)
    step_3.section_3.array.number = IntegerField("#### Number", flex=50 )
    # step_4 = Step("Load Growth Factor", description="Create scenarios based on the growrates", views=["get_plotly_view_1"])
    # step_4.section_1 = Section("Load Growth Factor", description="Specify the Load Growth Factor per customer group for the substation")
    # step_4.section_1.select_substation = OptionField("Substation", options=list_substations(), flex=50)
    # step_4.section_1.dynamic_array_1 = DynamicArray("Growrate")
    # step_4.section_1.dynamic_array_1.customer_type = OptionField("Customer Group", flex=50, description="Select the previously define customer group from the database to apply a load growth factor to.", options=list_customer_profiles())
    # step_4.section_1.dynamic_array_1.grow = NumberField("Load Growth Factor", flex=50, description="Growrate of the number of customers")
    # #step_4.section_1.dynamic_array_1.grow_type = OptionField("Type", flex=33, description="Growth type - percentage (baseline 100%) or absolute (number of connections)", options=['%', 'Absolute'], default='%')
    

class Controller(ViktorController):
    label = "My Entity Type"
    parametrization = Parametrization

    def train_model(self, params, **kwargs):
        upload_file = params["step_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        df = df.dropna()
        X = df[params["step_0"]["tab_train"]["features"]]
        y = df[params["step_0"]["tab_train"]["target"][0]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["step_0"]["tab_train"]["testset"], random_state=101)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_name = params["step_0"]["tab_train"]["model_name"]
        
        with open("models/{}.pkl".format(model_name), "wb") as f:
            pickle.dump(model, f)
        
        db.add_model(
            
            params["step_0"]["tab_train"]["model_name"], 
            list(X.columns),
            list(model.coef_),
            list(y_test),
            list(predictions),
            mean_squared_error(y_test, predictions), 
            mean_absolute_error(y_test, predictions))
        
        print(predictions)

    def add_substation(self, params, **kwargs):
        name = params['step_1']['section_1']['substation_name']
        #power_rating = params['step_1']['section_1']['substation_power']
        #num_feeders = params['step_1']['section_1']['number_of_feeders']
        location = (params['step_1']['section_1']['substation_location'].lat, params['step_1']['section_1']['substation_location'].lon)
        db.Substation(name, location).save_substation()

    def add_ami(self, params, **kwargs):
        ami_id = params['step_1']['section_2']['ami_id']
        location = (params['step_1']['section_2']['ami_location'].lat, params['step_1']['section_2']['ami_location'].lon)
        customer_type = params['step_1']['section_2']['customer_type']
        substation = params['step_1']['section_2']['substation']
        db.AMI(ami_id, location, customer_type, substation).save_AMI()

    def remove_substation(self, params, **kwargs):
        name = params['step_1']['section_3']['substation_name']
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

    @TableView("[I] Overview Input", duration_guess=20)
    def get_table_view(self, params, **kwargs):
        if params["step_0"]["tab_train"]["file"].file:
            upload_file = params["step_0"]["tab_train"]["file"].file
            data_file = BytesIO(upload_file.getvalue_binary())
            df = pd.read_csv(data_file)
            return TableResult(df)
    
    @DataView("[I] Model Performance Overview", duration_guess=20)
    def get_data_view(self, params, **kwargs):
        data = db.open_models()
        data = data['models']
        data_items = []
        for i in data:
            data_items.append(
                DataItem(i['model_name'], ' ', subgroup = DataGroup(
                    DataItem('MSE', i['MSE']),
                    DataItem('MAE', i['MAE']),
                    DataItem('Features \n \n', i['features'])
                ))
            )
        models = DataGroup(DataItem('Models', '', subgroup = DataGroup(*data_items)))
        return DataResult(models)
    
    @PlotlyView('[II] Prediction Analysis', duration_guess=10)
    def get_predict_view(self, params, **kwargs):
        
        model_name = params["step_0"]["tab_evaluate"]["model_name"]

        with open("models/{}.pkl".format(model_name), "rb") as f:
            model = pickle.load(f)
        
        upload_file = params["step_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        df = df.dropna()
        
        data = db.open_models()
        data = data['models']
        for m in data:
            if m['model_name'] == model_name:
                model_features = m['features']
                
                # TODO: add target as model attribute when uploading model
                model_target = 'Y house price of unit area'
        
        df_23_ytest = df[df['Time'].str.contains('22|23')][model_target]
        df_23_Xtest = df[df['Time'].str.contains('22|23')][model_features]

        x_ax = df[df['Time'].str.contains('22|23')]['Time']
        
        predictions = model.predict(df_23_Xtest)
        
        print(list(df_23_ytest))
        print(predictions)
            
        data = []
        data.append(go.Line(y=df_23_ytest, x=x_ax, name = 'Actual Data', line=dict(color='lightgrey', width=3)))
        data.append(go.Line(y=predictions, x=x_ax, name = 'Predicted Values', line=dict(color='royalblue', width=3)))

        fig = go.Figure(data = data)
        fig.update_layout(plot_bgcolor='whitesmoke', hovermode = 'x')
        fig = fig.to_json()
        
        return PlotlyResult(fig)
    
    @PlotlyView('[III] Forecast', duration_guess=10)
    def get_forecast_view(self, params, **kwargs):
        
        model_name = params["step_0"]["tab_forecast"]["model_name"]

        with open("models/{}.pkl".format(model_name), "rb") as f:
            model = pickle.load(f)
        
        upload_file = params["step_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        
        data = db.open_models()
        data = data['models']
        for m in data:
            if m['model_name'] == model_name:
                model_features = m['features']
                
                # TODO: add target as model attribute when uploading model
                model_target = 'Y house price of unit area'
        
        df_24_x_pred = df[df['Time'].str.contains('24')][model_features][6:]
        old_values = list(df[df['Time'].str.contains('23')][model_target])[6:]
        predictions = model.predict(df_24_x_pred)

        result = []
        for i in range(len(old_values)):
            result.append(round(((predictions[i] - old_values[i])/old_values[i])*100))
        
        x_ax = df[df['Time'].str.contains('24')]['Time'][len(df_24_x_pred):]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
 
        fig.add_trace(
        go.Bar(y=result, x=x_ax, name="LG %", opacity=0.5, marker=dict(color='lightseagreen')),
        secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(y=[round(mean(result))]*len(old_values), x=x_ax, mode = 'lines', name="Avg. LG %", line=dict(color='darkorange', width = 4)),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(y=old_values, x=x_ax, name="Old Data [kW]", line=dict(color='lightslategrey', width=3)),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(y=predictions, x=x_ax, name="Prediction [kW]", line=dict(color='royalblue', width=3)),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            plot_bgcolor = "whitesmoke"
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Load Growth Factor</b> %", secondary_y=False)
        fig.update_yaxes(title_text="<b>Peak Load</b> kW", secondary_y=True)
        fig.update_layout(hovermode="x")
        fig = fig.to_json()
        
        return PlotlyResult(fig)
    
    
    @PlotlyView('Selected Base Load Profile', duration_guess=1)
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

    @PlotlyView('New Base Load Profile', duration_guess=1)
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
    
    @GeoJSONView('Energy Asset Overview', duration_guess=10)
    def get_map_view_1(self, params, **kwargs):

        data = db.open_database()

        # Show all substations
        features = []

        for substation in data['substations']:
            features.append(substation)
        for AMI in data['AMIs']:
            features.append(AMI)

        geojson = {"type": "FeatureCollection",
                   "features": features}
        
        return GeoJSONResult(geojson)

        return MapResult(features)
    
    @PlotlyView('Aggregated Load Profile', duration_guess=10)
    def get_plotly_view_2(self, params, **kwargs):
        import plotly.graph_objects as go

        substation_name = params['step_3']['section_1']['substation_name']
        
        substation = db.Substation.get_substation_by_name(substation_name)
        if substation is None:
            raise UserError(f"Substation {substation_name} not found")

        # Get all AMIs connected to the selected substation
        connected_amis = substation.get_connected_AMIs()

        # Initialize a dictionary to store aggregated load per customer type
        aggregated_profiles = {}

        # List of all time intervals (assuming they are the same for all AMIs)
        time_intervals = [entry['time'] for entry in connected_amis[0]['properties']['load']['time_array']]

        # Initialize aggregated profiles with zeros for each customer type
        for ami in connected_amis:
            customer_type = ami['properties']['customer_type']
            if customer_type not in aggregated_profiles:
                aggregated_profiles[customer_type] = [0] * len(time_intervals)

            # Aggregate the loads
            for i, entry in enumerate(ami['properties']['load']['time_array']):
                aggregated_profiles[customer_type][i] += entry['value']

        # Now, we create the stacked bar chart
        fig = go.Figure()

        # Add each customer type to the stacked bar chart
        for customer_type, load_values in aggregated_profiles.items():
            fig.add_trace(go.Bar(
                x=time_intervals,
                y=load_values,
                name=customer_type
            ))

        # Customize layout
        fig.update_layout(
            barmode='stack',
            title=f'Aggregated Load Profile for Substation {substation_name}',
            xaxis_title='Time',
            yaxis_title='Load (kW)',
            legend_title='Customer Type'
        )

        fig = fig.to_json()

        return PlotlyResult(fig)
