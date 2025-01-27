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
    OutputField,
    GeoPolygonField,
    BooleanField,
    LineBreak,
    MultiSelectField, FileField, ActionButton, SetParamsButton, Page, MapSelectInteraction
)
from pathlib import Path
from io import BytesIO
from viktor import ViktorController, Color
import os

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
# import db_helpers as db
# import load_profiles as ph

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing
import numpy
import pickle
from plotly.subplots import make_subplots
from statistics import mean

from thpl.database import DatabaseConnectionManager
from thpl.energy_assets import Substation, Meter, Transformer
from thpl.sim_objects import LoadProfile, BaseProfile
from datetime import datetime


api_key = os.getenv("MONGO_API_WEB")
base_url = "https://eu-west-2.aws.data.mongodb-api.com/app/data-vjefvhb/endpoint/data/v1"

asset_connection = DatabaseConnectionManager.get_connector(
    connector_type='mongodb',
    base_url=base_url,
    db_name="geoWEB_demo",
    collection_name="geoWEB_demo_assets",
    data_source="geoWEB",
    headers={
        "Content-Type": "application/json",
        "Access-Control-Request-Headers": "*",
        "api-key": api_key,
    },
)

sim_obj_connection = DatabaseConnectionManager.get_connector(
    connector_type='mongodb',
    base_url=base_url,
    db_name="geoWEB_demo",
    collection_name="geoWEB_demo_simobjects",
    data_source="geoWEB",
    headers={
        "Content-Type": "application/json",
        "Access-Control-Request-Headers": "*",
        "api-key": api_key,
    })

# database.DatabaseManager.initialize(asset_connection)


# my_connection = database.DatabaseManager.get_instance()


def list_substations(**kwargs):
    """Get list of all substation names."""
    # db = DatabaseManager.get_instance()
    substations = asset_connection.find_many({
        'properties.type': 'substation'
    })
    return [sub['properties']['id'] for sub in substations] or ['No substations']

def list_connected_loads(params, **kwargs):
    substation_name = params.page_3.section_2.substation_name
    trafo = Transformer.by_id(substation_name)

    if trafo is not None:
        meter_objs, line_objs = trafo.get_connections()  
        return [meter.asset_id for meter in meter_objs] 
    
    return []

def list_base_profiles(**kwargs):
    result = sim_obj_connection.find_many({'type': 'BaseProfile'})
    result = list(set([entry['profile_id'] for entry in result]))
    return result

def list_customer_profiles(**kwargs):
    result = sim_obj_connection.find_many({'type': 'BaseProfile'})
    result = list(set([entry['customer_type'] for entry in result]))
    return result

def list_column_names(params, **kwargs):
    if params["page_0"]["tab_train"]["file"]:
        upload_file = params["page_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        return list(df.columns)
    else:
        return []

def create_default_content():
    result = sim_obj_connection.find_many({'type': 'LoadProfile'})
    default_content = []
    result = result[1]['profile']
    for key, value in result.items():
        default_content.append({'time': key, 'value': value})
    return default_content
    
def aggregated_load_profiles(params, **kwargs):
    """
    Aggregates all profiles (customer, solar, EV charging, and load growth) into a single dictionary.
    """
    time_intervals, customer_profiles = customer_load_profiles(params)
    solar_profile = solar_profiles(params)
    ev_profile = EV_charging_profiles(params)
    potential_load = customer_load_growth(params)

    aggregated_profiles = {"time_array": time_intervals}

    aggregated_profiles.update(customer_profiles)
    aggregated_profiles.update(ev_profile)
    aggregated_profiles.update(potential_load)
    aggregated_profiles.update(solar_profile)
    return aggregated_profiles


def EV_charging_profiles(params, **kwargs):
    """
    Generates EV charging profiles (slow, fast, and ultra-fast) based on the provided parameters.
    """
    data = params["page_3"]["section_3"]["array"]
    aggregated_profiles = {}

    for entry in data:
        number_of_chargers = entry["number"]
        power = 0

        if entry["type"] == "Slow (7 KW)":
            power = 7
            slow_profile = LoadProfile("Slow", power, "EV - Daily Charge - 5 PM")
            aggregated_profiles["Slow Charging"] = slow_profile.scale_profile().get_loadlist()
            aggregated_profiles["Slow Charging"] = [
                value * number_of_chargers for value in aggregated_profiles["Slow Charging"]
            ]

        elif entry["type"] == "Public Fast (22 KW)":
            power = 22
            fast_profile = LoadProfile("Fast", power, "Public Fast Charger")
            aggregated_profiles["Fast Charging"] = fast_profile.scale_profile().get_loadlist()
            aggregated_profiles["Fast Charging"] = [
                value * number_of_chargers for value in aggregated_profiles["Fast Charging"]
            ]

        elif entry["type"] == "Public Ultra Fast (70 KW)":
            power = 70
            ultra_fast_profile = LoadProfile("Ultra Fast", power, "Public Ultra Fast Charger")
            aggregated_profiles["Ultra Fast Charging"] = ultra_fast_profile.scale_profile().get_loadlist()
            aggregated_profiles["Ultra Fast Charging"] = [
                value * number_of_chargers for value in aggregated_profiles["Ultra Fast Charging"]
            ]

    return aggregated_profiles


def customer_load_profiles(params, **kwargs):
    """
    Aggregates customer load profiles from connected meters.
    """
    substation_name = params["page_3"]["section_0"]["substation_name"]
    trafo = Transformer.by_id(asset_connection, substation_name)
    meter_objs, _ = trafo.get_connections(asset_connection)

    aggregated_profiles = {}
    load_profile_list = [LoadProfile.from_meter(profile_id=meter.connection_type, database=sim_obj_connection, power_rating=meter.get_power_rating(), customer_type=meter.connection_type) for meter in meter_objs]

    time_intervals = list(load_profile_list[0].profile.keys())

    for profile in load_profile_list:
        if profile.profile_id not in aggregated_profiles:
            aggregated_profiles[profile.profile_id] = profile.get_loadlist()
        else:
            aggregated_profiles[profile.profile_id] = [
                x + y for x, y in zip(aggregated_profiles[profile.profile_id], profile.get_loadlist())
            ]
    return time_intervals, aggregated_profiles


def customer_load_growth(params, **kwargs):
    """
    Calculates potential customer load growth.
    """
    data = params["page_3"]["section_1"]["table"]
    time_intervals, customers_aggregated_load = customer_load_profiles(params)
    potential_load = {}

    for entry in data:
        customer_type = entry["customer_type"]
        customer_key = f"Load Growth - {customer_type}"
        load_growth_factor = entry["lgf"] * 0.01

        potential_load[customer_key] = [
            value * load_growth_factor for value in customers_aggregated_load[customer_type]
        ]

    return potential_load


def solar_profiles(params, **kwargs):
    """
    Generates solar profiles based on the transformer and provided parameters.
    """
    trafo_name = params["page_3"]["section_0"]["substation_name"]
    data = params["page_3"]["section_2"]["solar_array"]
    trafo = Transformer.by_id(asset_connection, trafo_name)

    if not data:
        return {"Solar": [0] * 96}

    aggregated_profiles = {"Solar": [0] * 96}

    total_peak_load = 0

    for entry in data:
        if not entry.get("percentage") or not entry.get("peak_load"):
            return {"Solar": [0] * 96}

        customer_group = entry["customer_group"]
        percentage = entry["percentage"] * 0.01
        peak_load_per_customer = entry["peak_load"]
        meter_objs, _ = trafo.get_connections(asset_connection)
        customer_count = sum(
            1 for meter in meter_objs if meter.connection_type == customer_group
        )
        total_peak_load += customer_count * percentage * peak_load_per_customer * 0.5

    total_peak_load = round(total_peak_load, 1)

    print(total_peak_load)
    solar_profile = LoadProfile.for_solar("residential", total_peak_load, sim_obj_connection)
    aggregated_profiles["Solar"] = [-value for value in solar_profile.get_loadlist()]

    return aggregated_profiles


def solar_peak_load(params, **kwargs):
    """
    Returns the peak load for solar profiles or a placeholder if unavailable.
    """
    solar_profile = solar_profiles(params)
    if solar_profile == {"Solar": [0] * 96}:
        return "Calculating..."
    else:
        return -min(solar_profile["Solar"])


class Parametrization(ViktorParametrization):
    # step_1 = Step("Manage Substations/Transformers", views=["get_map_view_1"], enabled=False)
    # step_1.section_1 = Section("Add Substations/Transformers")
    # step_1.section_1.intro = Text('Add a new substation to the database. Specify the name and location of the substation or transformer.')
    # step_1.section_1.substation_name = TextField('#### Name', flex = 100)
    # # step_1.section_1.substation_power = IntegerField('#### Powerrating', flex=33)
    # # step_1.section_1.number_of_feeders = IntegerField('#### Number of feeders', flex=33)
    # step_1.section_1.substation_location = GeoPointField('#### Location')
    # step_1.section_1.add_button = ActionButton('Add to Database', flex=100, method='add_substation')

    # step_1.section_2 = Section("Add Connection", description="Add a connection to the database")
    # step_1.section_2.connection_id = TextField('Conncetion ID', flex=50)
    # step_1.section_2.connection_location = GeoPointField('Location')
    # step_1.section_2.customer_type = OptionField('Customer Type', options=['Household', 'Industrial', 'Commercial'], flex=50)
    # step_1.section_2.substation = OptionField('Substation', options=list_substations(), flex=50)
    # step_1.section_2.add_button = ActionButton('Add to Database', flex=100, method='add_ami')

    # step_1.section_3 = Section("Remove Substations/Transformers", description="Remove")
    # step_1.section_3.substation_name = OptionField('Name', options=list_substations(), flex=50)
    # step_1.section_3.remove_button = ActionButton('Remove from Database', flex=50, method='remove_substation')
    # step_1.section_3.create_lines = ActionButton('Create lines', flex=50, method='create_lines')

    page_3 = Page("Energy Landscape Builder", views=['get_map_view_1', 'get_plotly_view_2', 'substation_load_overview'])
    
    page_3.section_0 = Section("Select Substation")
    page_3.section_0.substation_name = OptionField("Substation", options=['253166'], flex=50)

    page_3.section_01 =           Section("Assign Profiles")
    page_3.section_01.selection = SetParamsButton("Select Customers",
                                    method="select_customers",
                                    interaction=MapSelectInteraction('get_map_view_1', selection=['connections'], max_select=50), flex=100
                                    )
    
    page_3.section_01.substation_name = OptionField("Customer Profile", options=list_customer_profiles(), flex=50)

    page_3.section_1 = Section("Load Growth", description="Assign a load growth factor to each customer group for the selected substation.")
    page_3.section_1.text_1 = Text("""To carefully predict the load on the substation, a different load growth factor can be assigned to each customer group.""")
    page_3.section_1.table = DynamicArray("")
    page_3.section_1.table.customer_type = OptionField("Customer Group", options=list_customer_profiles(), flex=50)
    page_3.section_1.table.lgf = IntegerField("Load Growth (%)", description="Define how many customers of this type are connected.", flex=50)

    # page_3.section_1.connect_button = SetParamsButton('Connect', flex=100, method='connect_load')

    page_3.section_2 = Section("Rooftop Solar", description="Assign a peak power production load to substation, based on the number of solar panels installed.")
    page_3.section_2.intro = Text('This section allows you to add and simulate the impact of rooftop solar panels on the substation.')
    page_3.section_2.solar_array = DynamicArray("")
    page_3.section_2.solar_array.customer_group = OptionField('Customer Group', options=list_customer_profiles(), flex=50)
    page_3.section_2.solar_array.percentage = NumberField('% of Customers', description='Give the', flex=50, variant='slider', min=0, max=100, step=5)
    page_3.section_2.solar_array.peak_load = NumberField('Number of Panels (500 Wp)', flex=100)
    page_3.section_2.max_solar_load = OutputField('Installed Capacity', suffix='kWp', value=solar_peak_load,  flex=50)

    page_3.section_3 = Section("EV Charging", description="Add EV charging stations to the substation. Choose between the different types of charging stations and their corresponding chargin behaviour.")
    page_3.section_3.intro = Text('This section allows you to add and simulate the impact of EV charging stations on the substation.')
    page_3.section_3.array = DynamicArray("")
    page_3.section_3.array.type = OptionField("#### Type", options=['Slow (7 KW)', 'Public Fast (22 KW)', 'Public Ultra Fast (70 KW)'], flex=50)
    page_3.section_3.array.number = IntegerField("#### Number", flex=50 )
    
    ###
    
    step_2 = Page("Load Profile Manager", views=["get_plotly_view_1",'plotly_new_load_profile'])
    
    step_2.section_3 = Section("View Base Load Profile")
    step_2.section_3.select_load_profile = OptionField("Select Base Load Profile", options=list_base_profiles(), default='Residential 1', flex=50)
    
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

    ###
    
    # page_0 = Page("Load Growth Estimator", views = ["get_table_view", "get_data_view", "get_predict_view", "get_forecast_view"], width=30)
    
    # # TODO: Tab 0 Upload File
    # page_0.tab_train = Tab("[I] Train Model")
    # page_0.tab_train.file = FileField("Upload File", file_types=[".csv"], flex = 100)
    # page_0.tab_train.features = MultiSelectField('Select Features', options=list_column_names, flex=50)
    # page_0.tab_train.target = OptionField('Select Target', options=list_column_names, flex=50)
    # page_0.tab_train.testset = NumberField("Test Sample Size", min=0.2, max=0.5, step =0.1, variant='slider', flex =100)
    # page_0.tab_train.model_name = TextField("Model Name", flex=100)
    # page_0.tab_train.train = ActionButton("Train Model", method = 'train_model', flex=100)

    # page_0.tab_evaluate = Tab("[II] Evaluate Model")
    # page_0.tab_evaluate.model_name = TextField('Model Name', flex = 100)
    
    # page_0.tab_forecast = Tab("[III] Forecast")
    # page_0.tab_forecast.model_name = TextField('Model Name', flex = 100)


class Controller(ViktorController):
    label = "My Entity Type"
    parametrization = Parametrization

    def select_customers(self, params, event, **kwargs):
    
        selected_verdeelkasten = event.value        
        new_lines = []
        
        # for i in selected_verdeelkasten:
        #     kast = db.DB_find_entry(i)
        #     sel_dict = {
        #         "type":"Feature",
        #         "id":str(uuid.uuid4()),
        #         "geometry":{
        #             "type":"LineString",
        #             "coordinates":[ast.literal_eval(params.connect_tab.section_kasten.master_coord), kast['document']['geometry']['coordinates']]
        #             },
        #         "properties": {
        #             "from-to" : [kast['document']["id"], params.connect_tab.section_kasten.master_id],
        #             "description" : "master-connection",
        #             "stroke" : "#545454"
        #             }
        #         }
                
        #     new_lines.append(sel_dict)
        
        #     result = db.DB_add_many(new_lines)
        
        return SetParamsResult({})  
    
    def train_model(self, params, **kwargs):
        upload_file = params["page_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        df = df.dropna()

        X = df[params["page_0"]["tab_train"]["features"]]
        y = df[params["page_0"]["tab_train"]["target"]]


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["page_0"]["tab_train"]["testset"], random_state=101)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_name = params["page_0"]["tab_train"]["model_name"]
        
        with open("models/{}.pkl".format(model_name), "wb") as f:
            pickle.dump(model, f)
        
        db.add_model(
            params["page_0"]["tab_train"]["model_name"],
            params["page_0"]["tab_train"]["target"],
            list(X.columns),
            list(model.coef_),
            list(y_test),
            list(predictions),
            mean_squared_error(y_test, predictions), 
            mean_absolute_error(y_test, predictions)
            )
        
        print(predictions)

    def add_substation(self, params, **kwargs):
        name = params['step_1']['section_1']['substation_name']
        #power_rating = params['step_1']['section_1']['substation_power']
        #num_feeders = params['step_1']['section_1']['number_of_feeders']
        location = (params['step_1']['section_1']['substation_location'].lat, params['step_1']['section_1']['substation_location'].lon)
        db.Substation(name, location).save_substation()

    def add_ami(self, params, **kwargs):
        connection_id = params['step_1']['section_2']['connection_id']
        location = (params['step_1']['section_2']['connection_location'].lat, params['step_1']['section_2']['connection_location'].lon)
        customer_type = params['step_1']['section_2']['customer_type']
        substation = params['step_1']['section_2']['substation']
        db.AMI(connection_id, location, customer_type, substation).save_AMI()

    def remove_substation(self, params, **kwargs):
        name = params['step_1']['section_3']['substation_name']
        db.Substation.remove_substation(name)

    def create_lines(self, params, **kwargs):
        lines = db.create_lines()
        return

    def add_load_profile(self, params, **kwargs):
        data = params['step_2']['section_1']['dynamic_array_1']
        for profile in data:
            load = LoadProfile(profile['profile_name'], profile['peak_load'], profile['base_profile'])
            load.save_profile()

    def connect_load(self, params, **kwargs):
        substation_id = params['page_3']['section_0']['substation_name']
        data = params['page_3']['section_1']['dynamic_array_1']
        

        for connection in data:
            substation = my_connection.find_entry({'properties.id': substation_id})
            # substation = db.Substation.get_substation_by_name(substation_name)
            
            if substation is None:
                raise UserError(f"Substation {substation_id} not found")
            
            num_connections = connection['num_connections']
            load = ph.LoadProfile.find_load_profile(connection['customer_type'])
            
            # print(load.scaled_profile)
            substation.add_load(load, num_connections)
            
        result = SetParamsResult({
            "page_3": {
                "section_1": {
                    "dynamic_array_1": None
                }
            }
        })
        print(result)

    
        return result
        
    def remove_load(self, params, **kwargs):
        substation_name = params['page_3']['section_2']['substation_name']
        load_name = params['page_3']['section_2']['load_name']
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
        substation_name = params['page_3']['section_2']['substation_name']
        substation = db.Substation.get_substation_by_name(substation_name)
        return [load['name'] for load in substation.loads]

    @TableView("[I] Overview Input", duration_guess=20)
    def get_table_view(self, params, **kwargs):
        if params["page_0"]["tab_train"]["file"].file:
            upload_file = params["page_0"]["tab_train"]["file"].file
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
                DataItem("Model", "**{}**".format(i['model_name']), subgroup = DataGroup(
                    DataItem('Target', i['target']),
                    DataItem('Features', ' ', subgroup = DataGroup(*[DataItem(' ', x) for x in i['features']])),
                    DataItem('MSE', i['MSE']),
                    DataItem('MAE', i['MAE']),
                )
            ))
        models = DataGroup(*data_items)
        return DataResult(models)
    
    @PlotlyView('[II] Prediction Analysis', duration_guess=10)
    def get_predict_view(self, params, **kwargs):
        
        model_name = params["page_0"]["tab_evaluate"]["model_name"]

        with open("models/{}.pkl".format(model_name), "rb") as f:
            model = pickle.load(f)
        
        upload_file = params["page_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        df = df.dropna()
        
        data = db.open_models()
        data = data['models']
        for m in data:
            
            if m['model_name'] == model_name:
                model_features = m['features']
                model_target = m['target']
        
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
        
        model_name = params["page_0"]["tab_forecast"]["model_name"]

        with open("models/{}.pkl".format(model_name), "rb") as f:
            model = pickle.load(f)
        
        upload_file = params["page_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        
        data = db.open_models()
        data = data['models']
        for m in data:
            if m['model_name'] == model_name:
                model_features = m['features']
                model_target = m['target']
        
        df_select = df[df[model_target].isnull()]
        print(df_select)
        df_24_x_pred = df_select[model_features]
        print(df_24_x_pred)
        index_start = df_select.index[0]
        index_end = index_start + len(df_select)
        old_values = list(df[model_target])[index_start-12:index_end-12]
        predictions = model.predict(df_24_x_pred)

        result = []
        for i in range(len(old_values)):
            result.append(round(((predictions[i] - old_values[i])/old_values[i])*100))
        
        x_ax = df['Time'][index_start:index_end]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
 
        fig.add_trace(
        go.Bar(y=result, x=x_ax, name="LG %", opacity=0.3, marker=dict(color='lightsteelblue')),
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
            profile_name = 'Residential 1'
        
        profile_dict = sim_obj_connection.find_entry({'profile_id': profile_name, 'type': 'BaseProfile'})

        profile  = BaseProfile(profile_dict['profile_id'], profile_dict['profile'], profile_dict['customer_type'])
        

        time = [entry for entry in profile.profile_data.keys()]
        values = [entry for entry in profile.profile_data.values()]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=time, y=values, name='Load'))
        
        fig.update_layout(
            title= profile.profile_id + ' - Load Profile',
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
    
    @GeoJSONView('Energy Landscape Overview', duration_guess=2)
    def get_map_view_1(self, params, **kwargs):
        
        trafo_id = str(params.page_3.section_0.substation_name)
        query = {'properties.id': trafo_id}
        trafo = Transformer.by_id(asset_connection, trafo_id)
        # print(trafo['properties']['id'])

        # trafo = Transformer(
        #     asset_id=trafo['properties']['id'],
        #     creation_date=datetime.now(),
        #     location=(trafo['geometry']['coordinates'][0], trafo['geometry']['coordinates'][1]),
        #     voltage_in=1000,
        #     voltage_out=400,
        #     substation_id=123,
        #     rated_power =1000)
        
        meter_objs, line_objs = trafo.get_connections(asset_connection)
        features = []
        
        features.append(trafo.to_geojson())

        for meter in meter_objs:
            features.append(meter.to_geojson())

        for line in line_objs:
            features.append(line.to_geojson(asset_connection))

        

        # Show all substations
        # features = []

        # for substation in data['substations']:
        #     features.append(substation)
        # for AMI in data['AMIs']:
        #     features.append(AMI)
        # for line in data['lines']:
        #     features.append(line)

        # print(line)
        
        geojson = {"type": "FeatureCollection",
                   "features": features}
        

        return GeoJSONResult(geojson)
    
    @PlotlyView('Aggregated Load Profile', duration_guess=10)
    def get_plotly_view_2(self, params, **kwargs):
        import plotly.graph_objects as go
        aggregated_profiles = {}
        substation_name = params['page_3']['section_0']['substation_name']

        aggregated_profiles = aggregated_load_profiles(params)
        time_intervals = aggregated_profiles['time_array']
        aggregated_profiles.pop('time_array')

        # Define color scheme
        color_map = {
            'residential': 'rgb(4, 118, 208)',  # Blue for Household
            'Load Growth - residential': 'rgba(4, 118, 208, 0.5)',  # Lighter blue for potential growth
            'industrial': 'rgb(137, 207, 240)',  # DarkBlue for Industrial
            'Load Growth - industrial': 'rgba(137, 207, 240, 0.5)',  # Lighter for potential growth
            'commercial': 'rgb(0, 0, 139)',  # Orange for Commercial
            'Load Growth - commercial': 'rgba(0, 0, 139, 0.5)',  # Lighter orange for potential growth
            'Slow Charging': 'rgb(148, 103, 189)',  # Purple for Slow Charging
            'Fast Charging': 'rgba(148, 103, 189, 0.75)',  # Medium purple for Fast Charging
            'Ultra Fast Charging': 'rgba(148, 103, 189, 0.5)',  # Lighter purple for Ultra Fast Charging
            'Solar': 'rgb(255, 215, 0)'  # Yellow for Solar generation
        }

        # Define the order of layers
        layer_order = [
            'residential', 'Load Growth - residential',
            'commercial', 'Load Growth - commercial',
            'industrial', 'Load Growth - industrial',  # Bottom layers
            'Slow Charging', 'Fast Charging', 'Ultra Fast Charging',  # EV Charging
            'Solar'  # Solar on top
        ]

        # Create the stacked bar chart
        fig = go.Figure()

        # Add each customer type to the stacked bar chart in the defined order
        for customer_type in layer_order:
            if customer_type in aggregated_profiles:  # Check if key exists in aggregated_profiles
                fig.add_trace(go.Bar(
                    x=time_intervals,
                    y=aggregated_profiles[customer_type],
                    name=customer_type,
                    marker=dict(color=color_map.get(customer_type, 'gray'))  # Default to gray if no color found
                ))

        # Customize layout
        fig.update_layout(
            barmode='stack',
            title=f'Aggregated Load Profile for <b>{substation_name}',
            xaxis_title='Time',
            yaxis_title='Load (kW)',
            legend_title='Customer Type',
            template='plotly_white'  # Cleaner layout with white background
        )

        fig = fig.to_json()

        return PlotlyResult(fig)


    @DataView("Assumptions Overview", duration_guess=10)
    def substation_load_overview(self, params, **kwargs):
        substation_name = params['page_3']['section_0']['substation_name']
        aggregated_profiles = aggregated_load_profiles(params)
        aggregated_profiles.pop('time_array')

        solar_peakload = solar_peak_load(params)

        connected_amis = db.Substation.get_substation_by_name(substation_name).get_connected_AMIs()
        
        household_amis = len([ami for ami in connected_amis if ami['properties']['customer_type'] == 'Household'])
        industrial_amis = len([ami for ami in connected_amis if ami['properties']['customer_type'] == 'Industrial'])
        commercial_amis = len([ami for ami in connected_amis if ami['properties']['customer_type'] == 'Commercial'])
        total_amis = len(connected_amis)

        total_ev_power = 0

        for entry in params['page_3']['section_3']['array']:
            if entry['type'] == 'Slow (7 KW)':
                total_ev_power += 7 * entry['number']
            elif entry['type'] == 'Public Fast (22 KW)':
                total_ev_power += 22 * entry['number']
            elif entry['type'] == 'Public Ultra Fast (70 KW)':
                total_ev_power += 70 * entry['number']

        data = DataGroup(
            AMIs = DataItem('Connections to ' + substation_name, str(total_amis), subgroup=DataGroup(
                households=DataItem('Household Connections: (' + str(household_amis) +')', max(aggregated_profiles['Household']), prefix='Combined Peak Load', suffix='KW'),
                industrial=DataItem('Industrial Connections: (' + str(industrial_amis) +')', max(aggregated_profiles['Industrial']), prefix='Combined Peak Load', suffix='KW'),
                commercial=DataItem('Commercial Connections: (' + str(commercial_amis) +')', max(aggregated_profiles['Commercial']), prefix='Combined Peak Load', suffix='KW'),
            )),
            rooftop_solar = DataItem('Rooftop Solar', str(solar_peakload), suffix='kWp'),
            EV_charging = DataItem('EV Charging', '', subgroup=DataGroup(
                total_power=DataItem('Total installed power:', total_ev_power, suffix='kW'),
                number_of_chargers=DataItem('Total number of chargers:', sum([entry['number'] for entry in params['page_3']['section_3']['array']]), subgroup=DataGroup(
                    slow_chargers=DataItem('Slow (7 KW):', sum([entry['number'] for entry in params['page_3']['section_3']['array'] if entry['type'] == 'Slow (7 KW)'])),
                    fast_chargers=DataItem('Fast (22 KW):', sum([entry['number'] for entry in params['page_3']['section_3']['array'] if entry['type'] == 'Public Fast (22 KW)'])),
                    ultra_fast_chargers=DataItem('Ultra Fast (70 KW):', sum([entry['number'] for entry in params['page_3']['section_3']['array'] if entry['type'] == 'Public Ultra Fast (70 KW)']))
                )
                )
            ))
        )


        return DataResult(data)