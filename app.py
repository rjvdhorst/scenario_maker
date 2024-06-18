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
    DataView, DataResult, DataGroup, DataItem, PDFResult, PDFView
)

import plotly.graph_objects as go


class Parametrization(ViktorParametrization):
    _default_household_lp = [
        {'hour': '00:00', 'load': 0.5},  # Nighttime, reduced load
        {'hour': '01:00', 'load': 0.5},
        {'hour': '02:00', 'load': 0.5},
        {'hour': '03:00', 'load': 0.5},
        {'hour': '04:00', 'load': 0.5},
        {'hour': '05:00', 'load': 0.6},  # Waking up, lights, and small appliances
        {'hour': '06:00', 'load': 0.8},  # Morning routine, breakfast
        {'hour': '07:00', 'load': 1.2},  # Peak morning usage, cooking, and getting ready
        {'hour': '08:00', 'load': 1.2},
        {'hour': '09:00', 'load': 1.0},  # Daytime, reduced load
        {'hour': '10:00', 'load': 1.0},
        {'hour': '11:00', 'load': 1.0},
        {'hour': '12:00', 'load': 1.0},  # Lunchtime, some appliances
        {'hour': '13:00', 'load': 1.0},
        {'hour': '14:00', 'load': 1.0},
        {'hour': '15:00', 'load': 1.0},  # Afternoon, reduced load
        {'hour': '16:00', 'load': 0.8},  # Preparing dinner
        {'hour': '17:00', 'load': 1.0},  # Evening peak usage, cooking, and leisure
        {'hour': '18:00', 'load': 1.0},
        {'hour': '19:00', 'load': 0.8},  # Wind down, reduced load
        {'hour': '20:00', 'load': 0.7},
        {'hour': '21:00', 'load': 0.6},  # Evening, lights, and TV
        {'hour': '22:00', 'load': 0.6},
        {'hour': '23:00', 'load': 0.5}  # Nighttime, reduced load
    ]
        
    _default_ev_lp = [
        {'hour': '00:00', 'load': 0.5},  # Nighttime, reduced load
        {'hour': '01:00', 'load': 0.5},
        {'hour': '02:00', 'load': 0.5},
        {'hour': '03:00', 'load': 0.5},
        {'hour': '04:00', 'load': 0.5},
        {'hour': '05:00', 'load': 0.6},  # Waking up, lights, and small appliances
        {'hour': '06:00', 'load': 0.8},  # Morning routine, breakfast
        {'hour': '07:00', 'load': 1.2},  # Peak morning usage, cooking, and getting ready
        {'hour': '08:00', 'load': 1.2},
        {'hour': '09:00', 'load': 1.0},  # Daytime, reduced load
        {'hour': '10:00', 'load': 1.0},
        {'hour': '11:00', 'load': 1.0},
        {'hour': '12:00', 'load': 1.0},  # Lunchtime, some appliances
        {'hour': '13:00', 'load': 1.0},
        {'hour': '14:00', 'load': 1.0},
        {'hour': '15:00', 'load': 1.0},  # Afternoon, reduced load
        {'hour': '16:00', 'load': 0.8},  # Preparing dinner
        {'hour': '17:00', 'load': 1.0},  # Evening peak usage, cooking, and leisure
        {'hour': '18:00', 'load': 1.0},
        {'hour': '19:00', 'load': 0.8},  # Wind down, reduced load
        {'hour': '20:00', 'load': 0.7},
        {'hour': '21:00', 'load': 0.6},  # Evening, lights, and TV
        {'hour': '22:00', 'load': 0.6},
        {'hour': '23:00', 'load': 0.5}  # Nighttime, reduced load
    ]
    
    _default_business_lp = [
        {'hour': '00:00', 'load': 0.5},  # Nighttime, reduced load
        {'hour': '01:00', 'load': 0.5},
        {'hour': '02:00', 'load': 0.5},
        {'hour': '03:00', 'load': 0.5},
        {'hour': '04:00', 'load': 0.5},
        {'hour': '05:00', 'load': 0.6},  # Waking up, lights, and small appliances
        {'hour': '06:00', 'load': 0.8},  # Morning routine, breakfast
        {'hour': '07:00', 'load': 1.2},  # Peak morning usage, cooking, and getting ready
        {'hour': '08:00', 'load': 1.2},
        {'hour': '09:00', 'load': 1.0},  # Daytime, reduced load
        {'hour': '10:00', 'load': 1.0},
        {'hour': '11:00', 'load': 1.0},
        {'hour': '12:00', 'load': 1.0},  # Lunchtime, some appliances
        {'hour': '13:00', 'load': 1.0},
        {'hour': '14:00', 'load': 1.0},
        {'hour': '15:00', 'load': 1.0},  # Afternoon, reduced load
        {'hour': '16:00', 'load': 0.8},  # Preparing dinner
        {'hour': '17:00', 'load': 1.0},  # Evening peak usage, cooking, and leisure
        {'hour': '18:00', 'load': 1.0},
        {'hour': '19:00', 'load': 0.8},  # Wind down, reduced load
        {'hour': '20:00', 'load': 0.7},
        {'hour': '21:00', 'load': 0.6},  # Evening, lights, and TV
        {'hour': '22:00', 'load': 0.6},
        {'hour': '23:00', 'load': 0.5}  # Nighttime, reduced load
    ]

    _default_solar_lp = [
        {'hour': '00:00', 'load': 0.5},  # Nighttime, reduced load
        {'hour': '01:00', 'load': 0.5},
        {'hour': '02:00', 'load': 0.5},
        {'hour': '03:00', 'load': 0.5},
        {'hour': '04:00', 'load': 0.5},
        {'hour': '05:00', 'load': 0.6},  # Waking up, lights, and small appliances
        {'hour': '06:00', 'load': 0.8},  # Morning routine, breakfast
        {'hour': '07:00', 'load': 1.2},  # Peak morning usage, cooking, and getting ready
        {'hour': '08:00', 'load': 1.2},
        {'hour': '09:00', 'load': 1.0},  # Daytime, reduced load
        {'hour': '10:00', 'load': 1.0},
        {'hour': '11:00', 'load': 1.0},
        {'hour': '12:00', 'load': 1.0},  # Lunchtime, some appliances
        {'hour': '13:00', 'load': 1.0},
        {'hour': '14:00', 'load': 1.0},
        {'hour': '15:00', 'load': 1.0},  # Afternoon, reduced load
        {'hour': '16:00', 'load': 0.8},  # Preparing dinner
        {'hour': '17:00', 'load': 1.0},  # Evening peak usage, cooking, and leisure
        {'hour': '18:00', 'load': 1.0},
        {'hour': '19:00', 'load': 0.8},  # Wind down, reduced load
        {'hour': '20:00', 'load': 0.7},
        {'hour': '21:00', 'load': 0.6},  # Evening, lights, and TV
        {'hour': '22:00', 'load': 0.6},
        {'hour': '23:00', 'load': 0.5}  # Nighttime, reduced load
    ]
    
    step_1 = Step("Select Substations", description="Edit substations", views=["get_map_view_1"])
    step_1.section_1 = Section("Add Substations", description="Add a new substations")
    step_1.section_1.intro = Text('INTRO')
    step_1.section_1.substation_name = TextField('#### Name', flex=33)
    step_1.section_1.substation_power = IntegerField('#### Powerrating', flex=33)
    step_1.section_1.number_of_feeders = IntegerField('#### Number of feeders', flex=33)
    step_1.section_1.substation_location = GeoPointField('#### Location')
    step_1.section_1.add_button = ActionButton('Add Substation to Database', flex=100, method='empt_func')

    step_1.section_2 = Section("Remove Substations", description="Remove")
    step_1.section_2.substation_name = OptionField('Name', options=['Substation 1'], flex=50)
    step_1.section_2.remove_button = ActionButton('Remove Substation from Database', flex=50, method='empt_func')

    step_2 = Step("Load Profile", description="Create load profiles for all different customer groups", views=["get_plotly_view_1"])

    step_2.section_1 = Section("Edit load profiles", description="Customize the specified load profiles. ")
    step_2.section_1.intro = Text("""## Load Profiles \n To edit the load profiles of the customer type, upload a .csv file for the corresponding load profile for the customer.
                            The loadprofile will be normalized between 0 and 1, to serve as baseline for the customer type. In the next step, the peak load will be used as scaling factor.
                            """)

    step_2.section_1.dynamic_array_1 = DynamicArray("### Create Customer Type")
    step_2.section_1.dynamic_array_1.profile_name = TextField("Name", flex=50)
    step_2.section_1.dynamic_array_1.interval_field = IntegerField("Interval", suffix='min', flex=50)
    step_2.section_1.dynamic_array_1.load_profile = FileField("Load Profile", file_types=['.csv'], flex=50)
    step_2.section_1.normalize_button = ActionButton('Normalize Loadprofiles and add to Database', flex=100, method='empt_func')
    
    step_2.section_2 = Section("Show load profile")
    step_2.section_2.introtext = Text("## Show Profiles \n Show the selected loadprofiles, to analyze the normalized profiles.")
    step_2.section_2.select_load_profile = OptionField('Select Profile', options=['Household', 'Business etc'], flex=100)
    
    step_3 = Step("Current Energy Landscape", description="Specify the current energy landscape", views=['get_map_view_1'])
    step_3.section_1 = Section("Add demand", description="Specify the demand side for the substation")
    step_3.section_1.text_1 = Text("""### Demand \n To define what different scenario's can look like, it is possible to specify the number of customers of different types. This includes EV charging places""")
    step_3.section_1.dynamic_array_1 = DynamicArray("Connections")
    step_3.section_1.dynamic_array_1.substation_name = OptionField("Substation", options=['Substation 1', 'Substation 2'], flex=50)
    step_3.section_1.dynamic_array_1.customer_type = OptionField("Customer Type", options=['Household 1', 'Business 1', 'EV - public'], flex=50)
    step_3.section_1.dynamic_array_1.num_connections = IntegerField("Number of connections", description="Define how many customers of this type are connected.")
    step_3.section_1.dynamic_array_1.peak_load = IntegerField("Peak Load", description="Define the peak load of the connection")
    step_3.section_1.dynamic_array_1.number_field_1 = NumberField("Variance (%)", description="Define the variance in the load profile.")

    step_4 = Step("Future Scenario", description="Create scenarios based on the growrates", views=["get_plotly_view_1"])
    step_4.section_1 = Section("Growrates", description="Specify the growrates per customer side for the substation")
    step_4.section_1.dynamic_array_1 = DynamicArray("Growrate")
    step_4.section_1.dynamic_array_1.customer_type = OptionField("Customer", flex=33, description="Select the previously define customers from the database", options=['Household 1', 'Business 1', 'EV - public'])
    step_4.section_1.dynamic_array_1.grow = NumberField("Growrate", flex=33, description="Growrate of the number of customers")
    step_4.section_1.dynamic_array_1.grow_type = OptionField("Type", flex=33, description="Growth type - percentage (baseline 100%) or absolute (number of connections)", options=['%', 'Absolute'], default='%')
    

class Controller(ViktorController):
    label = "My Entity Type"
    parametrization = Parametrization

    def empt_func(self, params, **kwargs):
        pass

    @PlotlyView('Household Customer', duration_guess=1)
    def get_plotly_view_1(self, params, **kwargs):
        _default_household_lp = [
            {'hour': '00:00', 'load': 0.5},  # Nighttime, reduced load
            {'hour': '01:00', 'load': 0.5},
            {'hour': '02:00', 'load': 0.5},
            {'hour': '03:00', 'load': 0.5},
            {'hour': '04:00', 'load': 0.5},
            {'hour': '05:00', 'load': 0.6},  # Waking up, lights, and small appliances
            {'hour': '06:00', 'load': 0.8},  # Morning routine, breakfast
            {'hour': '07:00', 'load': 1.2},  # Peak morning usage, cooking, and getting ready
            {'hour': '08:00', 'load': 1.2},
            {'hour': '09:00', 'load': 1.0},  # Daytime, reduced load
            {'hour': '10:00', 'load': 1.0},
            {'hour': '11:00', 'load': 1.0},
            {'hour': '12:00', 'load': 1.0},  # Lunchtime, some appliances
            {'hour': '13:00', 'load': 1.0},
            {'hour': '14:00', 'load': 1.0},
            {'hour': '15:00', 'load': 1.0},  # Afternoon, reduced load
            {'hour': '16:00', 'load': 0.8},  # Preparing dinner
            {'hour': '17:00', 'load': 1.0},  # Evening peak usage, cooking, and leisure
            {'hour': '18:00', 'load': 1.0},
            {'hour': '19:00', 'load': 0.8},  # Wind down, reduced load
            {'hour': '20:00', 'load': 0.7},
            {'hour': '21:00', 'load': 0.6},  # Evening, lights, and TV
            {'hour': '22:00', 'load': 0.6},
            {'hour': '23:00', 'load': 0.5}  # Nighttime, reduced load
        ]

        lp_profile = _default_household_lp
        hourly_dict = {}
        
        for i in range(24):
            if len(str(i)) == 1:
                key = '0' + str(i) + ':00'
                hourly_dict[key] = lp_profile[i]['load']
            else:
                key = str(i) + ':00'
                hourly_dict[key] = lp_profile[i]['load']

        fig = go.Figure()

        trace = list(hourly_dict.values())

        fig.add_trace(go.Scatter(x=list(range(24)), y=trace, mode='lines+markers'))
        
        fig.update_layout(
            title='Aggregated Load Profile',
            xaxis_title='Hour of the Day',
            yaxis_title='Power Load (kW)',
            xaxis=dict(tickmode='linear'),
        )

        fig = fig.to_json()
        
        return PlotlyResult(fig)
    
    
    @MapView('Energy Asset Overview', duration_guess=1)
    def get_map_view_1(self, params, **kwargs):
        return MapResult(features=[])
    
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
    
