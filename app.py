from viktor.parametrization import (
    ViktorParametrization,
    TextField,
    NumberField,
    FileField,
    Section,
    Step,
    Tab,
    Text,
    OptionField,
    IntegerField,
    DynamicArray,
    MultiSelectField, FileField, ActionButton
)

from viktor.views import (
    PlotlyView,
    PlotlyResult,
    DataView, DataResult, DataGroup, DataItem, TableView, TableResult, GeoJSONResult, GeoJSONView
)
from viktor import ViktorController

import plotly.graph_objects as go
import db_helpers as db


def list_substations(**kwargs):
    data = db.open_database()
    substations = data['substations']
    if not substations:
        return ['No substations']
    substation_names = [substation['properties']['name'] for substation in substations] # Extract the name from each substation
    return substation_names

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

    step_3 = Step("Develop Energy Landscape", views=['get_map_view_1', 'get_plotly_view_2'])
    step_3.section_1 = Section("Load growth factor", description="Assign a load growth factor to each customer group for the selected substation.")
    step_3.section_1.text_1 = Text("""To carefully predict the load on the substation, a different load growth factor can be assigned to each customer group.""")
    step_3.section_1.substation_name = OptionField("Substation", options=list_substations(), flex=50)
    step_3.section_1.table = DynamicArray("### Load Growth Factor \n Add load to the selected substation")
    step_3.section_1.table.customer_type = OptionField("Customer Group", options=['Household', 'Industrial', 'Commercial'], flex=50)
    step_3.section_1.table.lgf = IntegerField("Load Growth Factor (percentage)", description="Define how many customers of this type are connected.", flex=50)

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

    def save_base_profile(self, params, **kwargs):
        profile_name = params['step_2']['section_2']['profile_name']

        profile = params['step_2']['section_2']['table']
        time_array = []
        for entry in profile:
            time_array.append({'time': entry['time'], 'value': entry['value']})

        ph.BaseProfile.save_base_profile(time_array, profile_name)

    def connect_load(self, params, **kwargs):
        substation_name = params['step_3']['section_1']['substation_name']
        data = params['step_3']['section_1']['table']
        for connection in data:
            substation = db.Substation.get_substation_by_name(substation_name)
            
            if substation is None:
                raise UserError(f"Substation {substation_name} not found")
            
            num_connections = connection['num_connections']
            load = ph.LoadProfile.find_load_profile(connection['customer_type'])
            
            substation.add_load(load, num_connections)
            
        result = SetParamsResult({
            "step_3": {
                "section_1": {
                    "table": None
                }
            }
        })
        print(result)

        return result

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
                model_target = 'Y house price of unit area'
        
        df_23_ytest = df[df['Time'].str.contains('22|23')][model_target]
        df_23_Xtest = df[df['Time'].str.contains('22|23')][model_features]

        x_ax = df[df['Time'].str.contains('22|23')]['Time']
        
        predictions = model.predict(df_23_Xtest)
        
        print(list(df_23_ytest))
        print(predictions)
            
        data = []
        data.append(go.Line(y=df_23_ytest, x=x_ax, name='Actual Data', line=dict(color='lightgrey', width=3)))
        data.append(go.Line(y=predictions, x=x_ax, name='Predicted Values', line=dict(color='royalblue', width=3)))

        fig = go.Figure(data=data)
        fig.update_layout(plot_bgcolor='whitesmoke', hovermode='x')
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
            go.Scatter(y=[round(mean(result))]*len(old_values), x=x_ax, mode='lines', name="Avg. LG %", line=dict(color='darkorange', width=4)),
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
            plot_bgcolor="whitesmoke"
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Load Growth Factor</b> %", secondary_y=False)
        fig.update_yaxes(title_text="<b>Peak Load</b> kW", secondary_y=True)
        fig.update_layout(hovermode="x")
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
        for line in data['lines']:
            features.append(line)

        print(line)

        geojson = {"type": "FeatureCollection",
                   "features": features}
        
        return GeoJSONResult(geojson)
    
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
        for line in data['lines']:
            features.append(line)

        print(line)

        geojson = {"type": "FeatureCollection",
                   "features": features}
        
        return GeoJSONResult(geojson)
    
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
            title=f'Aggregated Load Profile for {substation_name}',
            xaxis_title='Time',
            yaxis_title='Load (kW)',
            legend_title='Customer Type'
        )

        fig = fig.to_json()

        return PlotlyResult(fig)
