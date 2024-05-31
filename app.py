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
)
from viktor import ViktorController
from viktor.views import (
    PlotlyView,
    MapView,
    PlotlyResult,
    MapResult,
    DataView, DataResult, DataGroup, DataItem,
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
    
    step_1 = Step("Load Profile", description="Create loadprofiles for different type of customers", views=["get_plotly_view_1", "get_plotly_view_2", "get_plotly_view_3", "get_plotly_view_4"])
    step_1.section_1 = Section("Edit load profiles", description="Customize the specified load profiles. ")

    step_1.section_1.table_household = Table("Household", description="Give the load profile for a standard household connection.", default=_default_household_lp)
    step_1.section_1.table_household.hour = TextField("Hour")
    step_1.section_1.table_household.load = NumberField("Load (KW)")

    step_1.section_1.table_business = Table("Business", description="Give the load profile for a business connection", default=_default_business_lp)
    step_1.section_1.table_business.hour = TextField("Date")
    step_1.section_1.table_business.load = NumberField("Load (KW)")

    step_1.section_1.table_ev = Table("EV", description="Give the loadprofile for a general EV charging curve", default=_default_ev_lp)
    step_1.section_1.table_ev.hour = TextField("Date")
    step_1.section_1.table_ev.load = NumberField("Load (KW)")

    step_1.section_1.table_solar = Table("Solar array", description="Give the specifics for a solar array.", default=_default_solar_lp)
    step_1.section_1.table_solar.hour = TextField("Date")
    step_1.section_1.table_solar.load = NumberField("Load (KW)")

    step_2 = Step("Energy Landscape", description="Specify the different energy resources", views=["get_map_view_1", 'overview_data'])
    step_2.section_1 = Section("Demand Side ", description="Specify the demand side of the energy landscape")
    step_2.section_1.text_1 = Text("""To define what different scenario's can look like, it is possible to specify the number of customers of different types. This includes EV charging places""")
    step_2.section_1.dynamic_array_1 = DynamicArray("Demand")
    step_2.section_1.dynamic_array_1.option_field_1 = OptionField("Type", options=['Household', 'Business', 'EV'])
    step_2.section_1.dynamic_array_1.integer_field_1 = IntegerField("Amount ", description="Define how many customers of this type are connected. ")
    step_2.section_1.dynamic_array_1.number_field_1 = NumberField("Variance (%)", description="Define the variance in the load profile in percentages. ")
    
    step_2.section_2 = Section("Generation Side", description="Specify the generation side of the energy landscape.")
    step_2.section_2.text_1 = Text("""Define the generation side of the energy landscape. This includes solar, but also oil, gas and hydro. """)
    step_2.section_2.dynamic_array_1 = DynamicArray("Generation", description="Add all generation parts")
    step_2.section_2.dynamic_array_1.option_field_1 = OptionField("Type", options=['Gas', 'Solar', 'Hydro'])
    step_2.section_2.dynamic_array_1.text_field_1 = TextField("Name")
    step_2.section_2.dynamic_array_1.power_rating = IntegerField("Power Rating", description="Define the maximum constant power rating.")
    step_2.section_2.dynamic_array_1.geo_point_field_1 = GeoPointField("Location")

    step_3 = Step("Scenario's", description="Create different scenarios depending on ...", views=["get_plotly_view_5"])
    step_3.text_1 = Text("""This is an analysis tool part etc etc""")


class Controller(ViktorController):
    label = "My Entity Type"
    parametrization = Parametrization

    @PlotlyView('Household Customer', duration_guess=1)
    def get_plotly_view_1(self, params, **kwargs):
        lp_profile = params.step_1.section_1.table_household
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
            title='Household Load Profile',
            xaxis_title='Hour of the Day',
            yaxis_title='Power Load (kW)',
            xaxis=dict(tickmode='linear'),
        )

        fig = fig.to_json()
        
        return PlotlyResult(fig)
    
    @PlotlyView('Business Customer', duration_guess=1)
    def get_plotly_view_2(self, params, **kwargs):
        lp_profile = params.step_1.section_1.table_business
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
            title='Business Load Profile',
            xaxis_title='Hour of the Day',
            yaxis_title='Power Load (kW)',
            xaxis=dict(tickmode='linear'),
        )

        fig = fig.to_json()
        return PlotlyResult(fig)
    
    @PlotlyView('EV Load', duration_guess=1)
    def get_plotly_view_3(self, params, **kwargs):
        lp_profile = params.step_1.section_1.table_ev
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
            title='EV Charging Load Profile',
            xaxis_title='Hour of the Day',
            yaxis_title='Power Load (kW)',
            xaxis=dict(tickmode='linear'),
        )

        fig = fig.to_json()
        return PlotlyResult(fig)
    
    @PlotlyView('Solar Array', duration_guess=1)
    def get_plotly_view_4(self, params, **kwargs):
        lp_profile = params.step_1.section_1.table_solar
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
            title='Solar Panel Profile',
            xaxis_title='Hour of the Day',
            yaxis_title='Power Load (kW)',
            xaxis=dict(tickmode='linear'),
        )

        fig = fig.to_json()
        return PlotlyResult(fig)
    
    @MapView('Energy Asset Overview', duration_guess=1)
    def get_map_view_1(self, params, **kwargs):
        return MapResult(features=[])
    
    @DataView('Overview', duration_guess=1)
    def overview_data(self, params, **kwargs):
        data = DataGroup(
            group_a=DataItem('Total Demand', '', subgroup=DataGroup(
                sub_group1=DataItem('Average Power Generation', 11, suffix='MW', subgroup=DataGroup(
                    value_a=DataItem('Households', 5, suffix='MW'),
                    value_b=DataItem('Business', 5, suffix='MW'),
                    value_c=DataItem('EV', 1, suffix='MW'))), 
                sub_group2=DataItem('Max Power Generation', 9, suffix='MW', subgroup=DataGroup(
                    value_a=DataItem('Households', 3, suffix='MW'),
                    value_b=DataItem('Business', 5, suffix='MW'),
                    value_c=DataItem('EV', 1, suffix='MW')),
            ))),
            group_b=DataItem('Total Generation', '', subgroup=DataGroup(
                sub_group1=DataItem('Average Power Generation', 10, suffix='MW', subgroup=DataGroup(
                    value_a=DataItem('Gas and Oil', 8, suffix='MW'),
                    value_b=DataItem('Solar', 2, suffix='MW', explanation_label='Total summation of all maximum solar, including home solar.'))),
                sub_group2=DataItem('Average Power Generation', 7.5, suffix='MW', subgroup=DataGroup(
                    value_a=DataItem('Gas and Oil', 6, suffix='MW'),
                    value_b=DataItem('Solar', 1.5, suffix='MW', explanation_label='Total summation of all maximum solar, including home solar.')
                ))
            ))
        )
        return DataResult(data)
    

    @PlotlyView('Analysis', duration_guess=1)
    def get_plotly_view_5(self, params, **kwargs):
        return PlotlyResult(figure=dict())
    
