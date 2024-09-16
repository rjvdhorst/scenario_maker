from viktor.parametrization import (
    ViktorParametrization,
    TextField,
    NumberField,
    FileField,
    Section,
    Tab,
    Text,
    OptionField,
    IntegerField,
    DynamicArray,
    OutputField,
    MultiSelectField, FileField, ActionButton, Page, GeoPointField, Table
)
from io import BytesIO
from viktor import ViktorController

from viktor.views import (
    PlotlyView,
    PlotlyResult,
    DataView, DataResult, DataGroup, DataItem, TableView, TableResult, GeoJSONResult, GeoJSONView
)

import plotly.graph_objects as go
# import db_helpers as db
# import load_profiles as ph

from power_grid import Database, Profiles, Entities, Utils, Models

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import pickle
from plotly.subplots import make_subplots
from statistics import mean


def list_substations(**kwargs):
    substations = Database().open_database()['substations']
    if not substations:
        return ['No substations']
    substation_names = [substation['properties']['name'] for substation in substations] 
    return substation_names

def list_base_profiles(**kwargs):
    return Profiles.LoadProfile.list_names()

def list_customer_profiles(**kwargs):
    customer_profiles = Profiles.LoadProfile.all_customer_profiles()
    return [profile.name for profile in customer_profiles]

def list_connected_loads(params, **kwargs):
    substation_name = params.page_3.section_2.substation_name
    substation = Entities.Substation.get_substation_by_name(substation_name)
    if substation is not None:
        return [load.name for load in substation.loads]
    else:
        return []

def list_column_names(params, **kwargs):
    if params["page_0"]["tab_train"]["file"]:
        upload_file = params["page_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        return list(df.columns)
    else:
        return []

def create_default_content():
    load_profile = Profiles.LoadProfile.find_load_profile('Industrial')
    default_content = load_profile.profile_dict()
    default_array = default_content['time_array']
    return 

def aggregated_load_profiles(params, **kwargs):
    time_intervals, customer_profiles = customer_load_profiles(params)
    solar_profile = solar_profiles(params)
    ev_profile = EV_charging_profiles(params)
    potential_load = customer_load_growth(params)
    aggregated_profiles = {}
    aggregated_profiles['time_array'] = time_intervals 
    aggregated_profiles.update(customer_profiles)
    aggregated_profiles.update(ev_profile)
    aggregated_profiles.update(potential_load)
    aggregated_profiles.update(solar_profile)
    
    return aggregated_profiles
    
def EV_charging_profiles(params, **kwargs):
    data = params['page_3']['section_3']['array']
    aggregated_profiles = {}

    for entry in data:
        number_of_chargers = entry['number']
        power = 0

        if entry['type'] == 'Slow (7 KW)':
            power = 7
            slow_profile = Profiles.LoadProfile('Slow', power, 'EV - Daily Charge - 5 PM').scale_profile()
            aggregated_profiles['Slow Charging'] = [entry['value']*number_of_chargers for entry in slow_profile['time_array']]
        
        elif entry['type'] == 'Public Fast (22 KW)':
            power = 22
            fast_profile = Profiles.LoadProfile('Fast', power, 'Public Fast Charger').scale_profile()
            aggregated_profiles['Fast Charging'] = [entry['value']*number_of_chargers for entry in fast_profile['time_array']]

        elif entry['type'] == 'Public Ultra Fast (70 KW)':
            power = 70
            ultra_fast_profile = Profiles.LoadProfile('Ultra Fast', power, 'Public Ultra Fast Charger').scale_profile()
            aggregated_profiles['Ultra Fast Charging'] = [entry['value']*number_of_chargers for entry in ultra_fast_profile['time_array']]
    return aggregated_profiles

def customer_load_profiles(params, **kwargs):
    substation_name = params['page_3']['section_1']['substation_name']
    substation = Entities.Substation.get_substation_by_name(substation_name)
    
    connections = substation.get_connected_connections()

    print(connections[0].load)
    aggregated_profiles = {}

    time_intervals = [entry['time'] for entry in connections[0].load['time_array']]
    
    for connection in connections:
        customer_type = connection.customer_type
        if customer_type not in aggregated_profiles:
            aggregated_profiles[customer_type] = [0] * len(time_intervals)

        for i, entry in enumerate(connection.load['time_array']):
            aggregated_profiles[customer_type][i] += entry['value']

    return time_intervals, aggregated_profiles


def customer_load_growth(params, **kwargs):
    data = params['page_3']['section_1']['table']

    time_intervals, customers_aggregated_load = customer_load_profiles(params)
    
    potential_load = {}

    for entry in data:
        customer_type = entry['customer_type']
        customer_key = 'Potential Growth - ' + customer_type
        load_growth_factor = entry['lgf']*0.01
        potential_load[customer_key] = [entry * load_growth_factor for entry in customers_aggregated_load[customer_type]]

    return potential_load


def solar_profiles(params, **kwargs):
    substation_name = params['page_3']['section_1']['substation_name']
    data = params['page_3']['section_2']['solar_array']
    substation = Entities.Substation.get_substation_by_name(substation_name)
    peak_load = 0

    for entry in data:
        customer_group = entry['customer_group']
        number_of_customers = 0
        percentage = entry['percentage']*0.01
        number_of_panels = entry['peak_load']

        for connection in substation.get_connected_connections():
            if connection.customer_type == customer_group:
                number_of_customers += 1
        
        print(number_of_customers)
        
        peak_load += number_of_customers * percentage * number_of_panels * 0.5

    peak_load = round(peak_load, 1)
    solar_profile = Entities.LoadProfile('Solar', peak_load, 'Solar_array').scale_profile()
    aggregated_profiles = {}
    aggregated_profiles['Solar'] = [0] * 96

    for i, entry in enumerate(solar_profile['time_array']):
        aggregated_profiles['Solar'][i] -= entry['value']

    if aggregated_profiles['Solar'] == [0]*96:
        return {}
    return aggregated_profiles

def solar_peak_load(params, **kwargs):
    if solar_profiles(params) == {}:
        return 0
    return -1*min(solar_profiles(params)['Solar'])

class Parametrization(ViktorParametrization):
    page_1 = Page("Manage Substations/Transformers", views=["get_map_view_1"])
    page_1.section_1 = Section("Add Substations/Transformers")
    page_1.section_1.intro = Text('Add a new substation to the database. Specify the name and location of the substation or transformer.')
    page_1.section_1.substation_name = TextField('#### Name', flex = 100)
    # page_1.section_1.substation_power = IntegerField('#### Powerrating', flex=33)
    # page_1.section_1.number_of_feeders = IntegerField('#### Number of feeders', flex=33)
    page_1.section_1.substation_location = GeoPointField('#### Location')
    page_1.section_1.add_button = ActionButton('Add to Database', flex=100, method='add_substation')

    page_1.section_2 = Section("Add AMI", description="Add an AMI to the database")
    page_1.section_2.ami_id = TextField('AMI ID', flex=50)
    page_1.section_2.ami_location = GeoPointField('Location')
    page_1.section_2.customer_type = OptionField('Customer Type', options=['Household', 'Industrial', 'Commercial'], flex=50)
    page_1.section_2.substation = OptionField('Substation', options=list_substations(), flex=50)
    page_1.section_2.add_button = ActionButton('Add to Database', flex=100, method='add_connection')

    page_1.section_3 = Section("Remove Substations/Transformers", description="Remove")
    page_1.section_3.substation_name = OptionField('Name', options=list_substations(), flex=50)
    page_1.section_3.remove_button = ActionButton('Remove from Database', flex=50, method='remove_substation')
    page_1.section_3.create_lines = ActionButton('Create lines', flex=50, method='create_lines')

    page_2 = Page("Manage Load Profiles", description="Manage load profiles for different customer groups", views=['plotly_new_load_profile', "get_plotly_view_1"])

    page_2.section_1 = Section("Create Customer Group", description="Customize the specified load profiles. ")
    page_2.section_1.intro = Text("""Create load profiles for different customer types. Specify the name, peak load, and base profile. The base profile is a predefined profile that can be scaled to the peak load.""")

    page_2.section_1.dynamic_array_1 = DynamicArray("")
    page_2.section_1.dynamic_array_1.profile_name = TextField("Name", flex=33)
    page_2.section_1.dynamic_array_1.peak_load = NumberField("Peak Load", suffix='KW', flex=33)
    page_2.section_1.dynamic_array_1.base_profile = OptionField("Base Load Profile", options=list_base_profiles(), flex=33)
    page_2.section_1.normalize_button = ActionButton('Add to database', flex=100, method='add_load_profile')
    
    page_2.section_2 = Section("Add Base Load Profiles")
    page_2.section_2.introtext = Text("A Base Load Profile can be configured by filling the table below. Note that the profile is a normalized profile. A value of 1 is equal to the peak load that will be assigned to the profile in a next step.")
    page_2.section_2.profile_name = TextField("##### Name", flex=80)
    page_2.section_2.table = Table("Create a New Base Load Profile", default=create_default_content())
    page_2.section_2.table.time = TextField('time')
    page_2.section_2.table.value = NumberField('value')
    page_2.section_2.upload_button = ActionButton("Save Base Load Profile", flex=60, method='save_base_profile')

    page_2.section_3 = Section("View Base Load Profile")
    page_2.section_3.select_load_profile = OptionField("Select Base Load Profile", options=list_base_profiles(), default='Household', flex=50)
    
    page_0 = Page("Load Growth Factor Regression", views = ["get_table_view", "get_data_view", "get_predict_view", "get_forecast_view"], width=30)
    
    # TODO: Tab 0 Upload File
    page_0.tab_train = Tab("[I] Train Model")
    page_0.tab_train.file = FileField("Upload File", file_types=[".csv"], flex = 100)
    page_0.tab_train.features = MultiSelectField('Select Features', options=list_column_names, flex=50)
    page_0.tab_train.target = OptionField('Select Target', options=list_column_names, flex=50)
    page_0.tab_train.testset = NumberField("Test Sample Size", min=0.2, max=0.5, step =0.1, variant='slider', flex =100)
    page_0.tab_train.model_name = TextField("Model Name", flex=100)
    page_0.tab_train.train = ActionButton("Train Model", method = 'train_model', flex=100)

    page_0.tab_evaluate = Tab("[II] Evaluate Model")
    page_0.tab_evaluate.model_name = TextField('Model Name', flex = 100)
    
    page_0.tab_forecast = Tab("[III] Forecast")
    page_0.tab_forecast.model_name = TextField('Model Name', flex = 100)
    

    page_3 = Page("Develop Energy Landscape", views=['get_map_view_1', 'get_plotly_view_2', 'substation_load_overview'])
    page_3.section_1 = Section("Load growth factor", description="Assign a load growth factor to each customer group for the selected substation.")
    page_3.section_1.text_1 = Text("""To carefully predict the load on the substation, a different load growth factor can be assigned to each customer group.""")
    page_3.section_1.substation_name = OptionField("Substation", options=list_substations(), flex=50)
    page_3.section_1.table = DynamicArray("### Load Growth Factor \n Add load to the selected substation")
    page_3.section_1.table.customer_type = OptionField("Customer Group", options=['Household', 'Industrial', 'Commercial'], flex=50)
    page_3.section_1.table.lgf = IntegerField("Load Growth Factor (percentage)", description="Define how many customers of this type are connected.", flex=50)


    page_3.section_2 = Section("Rooftop Solar", description="Assign a peak power production load to substation, based on the number of solar panels installed.")
    page_3.section_2.intro = Text('This section allows you to add and simulate the impact of rooftop solar panels on the substation.')
    page_3.section_2.solar_array = DynamicArray("Calculated Rooftop Solar")
    page_3.section_2.solar_array.customer_group = OptionField('Customer Group', options=['Household', 'Industrial', 'Commercial'], flex=50)
    page_3.section_2.solar_array.percentage = IntegerField('Percentage', description='Give the', flex=50)
    page_3.section_2.solar_array.peak_load = NumberField('Number of panels (500W each)', flex=100)
    page_3.section_2.max_solar_load = OutputField('Load', suffix='KW', value=solar_peak_load,  flex=50)

    page_3.section_3 = Section("EV charging landscape", description="Add EV charging stations to the substation. Choose between the different types of charging stations and their corresponding chargin behaviour.")
    page_3.section_3.intro = Text('This section allows you to add and simulate the impact of EV charging stations on the substation.')
    page_3.section_3.array = DynamicArray("EV Charging Stations")
    page_3.section_3.array.type = OptionField("#### Type", options=['Slow (7 KW)', 'Public Fast (22 KW)', 'Public Ultra Fast (70 KW)'], flex=50)
    page_3.section_3.array.number = IntegerField("#### Number", flex=50 )
    

class Controller(ViktorController):
    label = "My Entity Type"
    parametrization = Parametrization

    def train_model(self, params, **kwargs):
        # Extract uploaded file and read into pandas DataFrame
        upload_file = params["page_0"]["tab_train"]["file"].file
        data_file = BytesIO(upload_file.getvalue_binary())
        df = pd.read_csv(data_file)
        
        # Drop rows with any NaN values
        df = df.dropna()

        # Select features and target from DataFrame
        X = df[params["page_0"]["tab_train"]["features"]]
        y = df[params["page_0"]["tab_train"]["target"]]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["page_0"]["tab_train"]["testset"], random_state=101)
        
        # Initialize and train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Name of the model (used for saving)
        model_name = params["page_0"]["tab_train"]["model_name"]
        
        # Save the trained model using pickle
        with open("models/{}.pkl".format(model_name), "wb") as f:
            pickle.dump(model, f)
        
        db = Database()
        # Assuming db.add_model adds model details to a database or some storage
        db.add_model(
            model_name,
            params["page_0"]["tab_train"]["target"],
            list(X.columns),
            list(model.coef_),
            list(y_test),
            list(predictions),
            mean_squared_error(y_test, predictions), 
            mean_absolute_error(y_test, predictions)
        )
        return
    
    def add_substation(self, params, **kwargs):
        """
        Adds a new substation to the database with the provided parameters.

        Parameters:
        -----------
        params : dict
            A dictionary containing information about the substation to be added.
        kwargs : dict, optional
            Additional arguments that might be passed for future use.

        Returns:
        --------
        None
            The function saves the substation data to the database and does not return a value.
        """
        
        # Extracting substation name from params
        name = params['page_1']['section_1']['substation_name']
        
        # Extracting substation location as a tuple of (latitude, longitude)
        location = (params['page_1']['section_1']['substation_location'].lat, 
                    params['page_1']['section_1']['substation_location'].lon)
        
        # Creating a new substation entry in the database and saving it

        substation = Entities.Substation(name, location)
        Entities.Substation.save_substation(substation)

    def add_connection(self, params, **kwargs):
        """
        Adds a new meter to the database with the provided parameters.

        Parameters:
        -----------
        params : dict
            A dictionary containing information about the substation to be added.
        kwargs : dict, optional
            Additional arguments that might be passed for future use.

        Returns:
        --------
        None
            The function saves the substation data to the database and does not return a value.
        """
        
        connection_id = params['page_1']['section_2']['ami_id']
        location = (params['page_1']['section_2']['ami_location'].lat, params['page_1']['section_2']['ami_location'].lon)
        customer_type = params['page_1']['section_2']['customer_type']
        substation = params['page_1']['section_2']['substation']
        connection = Entities.Connection(connection_id, location, customer_type, substation)
        Entities.Connection.save_connection(connection)


    def remove_substation(self, params, **kwargs):
        """
        Removes a substation from the database.

        Args:
            params (dict): A dictionary containing the parameters for removing the substation.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        name = params['page_1']['section_3']['substation_name']
        Entities.Substation.remove_substation(name)

    def create_lines(self, params, **kwargs):
        """
        Creates the powerlines from meter to substation.

        Args:
            params: The parameters for creating the lines.
            kwargs: Additional keyword arguments.

        Returns:
            None
        """
        lines = Utils.create_lines()
        return

    def add_load_profile(self, params, **kwargs):
        data = params['page_2']['section_1']['dynamic_array_1']
        for profile in data:
            load = Profiles.LoadProfile(profile['profile_name'], profile['peak_load'], profile['base_profile'])
            load.save_profile()

        
    def save_base_profile(self, params, **kwargs):
        profile_name = params['page_2']['section_2']['profile_name']

        profile = params['page_2']['section_2']['table']
        time_array = []
        for entry in profile:
            time_array.append({'time': entry['time'], 'value': entry['value']})

        Profiles.BaseProfile.save_base_profile(time_array, profile_name)


    @staticmethod
    def list_connected_loads(params, **kwargs):
        substation_name = params['page_3']['section_2']['substation_name']
        substation = Entities.Substation.get_substation_by_name(substation_name)
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
        db = Database()
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
        
        data = Database().open_models()
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
        
        data = Database().open_models()
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
        profile_name = params['page_2']['section_3']['select_load_profile']

        if profile_name == None:
            profile_name = 'Household'
        
        print(profile_name)
        profile  = Profiles.BaseProfile(profile_name).load_profile_data()
        print(profile)
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
        profile_name = params['page_2']['section_3']['select_load_profile']

        if profile_name == None:
            profile_name = 'Industrial'
        
        profile  = params['page_2']['section_2']['table']

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
        db = Database()
        data = db.open_database()

        # Show all substations
        features = []

        for substation in data['substations']:
            features.append(substation)
        for connection in data['connections']:
            features.append(connection)
        for line in data['lines']:
            features.append(line)

        print(line)

        geojson = {"type": "FeatureCollection",
                   "features": features}
        
        return GeoJSONResult(geojson)
    
    @PlotlyView('Aggregated Load Profile', duration_guess=10)
    def get_plotly_view_2(self, params, **kwargs):
        import plotly.graph_objects as go
        aggregated_profiles = {}
        substation_name = params['page_3']['section_1']['substation_name']

        aggregated_profiles = aggregated_load_profiles(params)
        time_intervals = aggregated_profiles['time_array']
        aggregated_profiles.pop('time_array')

        # Define color scheme
        color_map = {
            'Household': 'rgb(31, 119, 180)',  # Blue for Household
            'Potential Growth - Household': 'rgba(31, 119, 180, 0.5)',  # Lighter blue for potential growth
            'Industrial': 'rgb(44, 160, 44)',  # Green for Industrial
            'Potential Growth - Industrial': 'rgba(44, 160, 44, 0.5)',  # Lighter green for potential growth
            'Commercial': 'rgb(255, 127, 14)',  # Orange for Commercial
            'Potential Growth - Commercial': 'rgba(255, 127, 14, 0.5)',  # Lighter orange for potential growth
            'Slow Charging': 'rgb(148, 103, 189)',  # Purple for Slow Charging
            'Fast Charging': 'rgba(148, 103, 189, 0.75)',  # Medium purple for Fast Charging
            'Ultra Fast Charging': 'rgba(148, 103, 189, 0.5)',  # Lighter purple for Ultra Fast Charging
            'Solar': 'rgb(255, 255, 0)'  # Yellow for Solar generation
        }

        # Define the order of layers
        layer_order = [
            'Household', 'Potential Growth - Household',  # Bottom layers
            'Industrial', 'Potential Growth - Industrial',
            'Commercial', 'Potential Growth - Commercial',
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
            title=f'Aggregated Load Profile for {substation_name}',
            xaxis_title='Time',
            yaxis_title='Load (kW)',
            legend_title='Customer Type',
            template='plotly_white'  # Cleaner layout with white background
        )

        fig = fig.to_json()

        return PlotlyResult(fig)




    @DataView("Substation - Load overview", duration_guess=10)
    def substation_load_overview(self, params, **kwargs):
        substation_name = params['page_3']['section_1']['substation_name']
        aggregated_profiles = aggregated_load_profiles(params)
        aggregated_profiles.pop('time_array')

        solar_peakload = solar_peak_load(params)

        connected_amis = Entities.Substation.get_substation_by_name(substation_name).get_connected_AMIs()
        
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
            rooftop_solar = DataItem('Rooftop Solar', 'Maximum peak generation: ' + str(solar_peakload), suffix='KW'),
            EV_charging = DataItem('EV Charging', '', subgroup=DataGroup(
                total_power=DataItem('Total installed power:', total_ev_power, suffix='KW'),
                number_of_chargers=DataItem('Total number of chargers:', sum([entry['number'] for entry in params['page_3']['section_3']['array']]), subgroup=DataGroup(
                    slow_chargers=DataItem('Slow (7 KW):', sum([entry['number'] for entry in params['page_3']['section_3']['array'] if entry['type'] == 'Slow (7 KW)'])),
                    fast_chargers=DataItem('Fast (22 KW):', sum([entry['number'] for entry in params['page_3']['section_3']['array'] if entry['type'] == 'Public Fast (22 KW)'])),
                    ultra_fast_chargers=DataItem('Ultra Fast (70 KW):', sum([entry['number'] for entry in params['page_3']['section_3']['array'] if entry['type'] == 'Public Ultra Fast (70 KW)']))
                )
                )
            ))
        )


        return DataResult(data)