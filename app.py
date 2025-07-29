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

import viktor as vkt

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
import numpy as np
import pickle
from plotly.subplots import make_subplots
from statistics import mean

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json


import os
import json
from openai import OpenAI

#OPENAI_API_KEY = "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"
#OPENAI_API_KEY = "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"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def open_database():
    json_file = Path(__file__).parent / 'dataLLM.json'

    with open(json_file, 'r') as f:
        data = json.load(f)
        
    return data

def save_database(data):
    json_file = Path(__file__).parent / 'dataLLM.json'

    with open(json_file, 'w') as f:
        json.dump(data, f)

# Define the output schema and parser
class EnergyScenario(BaseModel):
    households: int = Field(..., description="Number of households")
    commercials: int = Field(..., description="Number of commercial buildings")
    industrials: int = Field(..., description="Number of industrial customers")
    chargers: int = Field(..., description="Number of EV chargers")
    solar_capacity_kw: int = Field(..., description="Estimated total solar capacity in kW")
    charging_start: str = Field(..., description="Typical charging session start time, e.g. '18:00'")

parser = JsonOutputParser(pydantic_object=EnergyScenario)

# LLM setup
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Prompt to extract scenario values
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that extracts energy scenario parameters from a user's description."),
    ("human", "{input}"),
    ("system", "Based on this input, output the corresponding energy scenario dictionary using this format:\n{format_instructions}")
])

chain = extraction_prompt | llm | parser

# Prompt to summarize assumptions
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that creates a scenario summary for an energy planner based on the parameters below."),
    ("human", "This is the final dictionary: {dict}.\nPlease write a short summary describing the neighborhood and the assumptions that were made.")
])

summary_chain = summary_prompt | llm

class Parametrization(ViktorParametrization):

    page_LLM = Page("Scenario Development Demo CARILEC", views = ["get_plotly_view_2"])
    page_LLM.chat = vkt.Chat("Chatbot", method="call_llm", first_message="Hi, here to support you in developing an energy scenario! Please describe the neighborhood you want to develop a load profile for.")


class Controller(ViktorController):
    label = "My Entity Type"
    parametrization = Parametrization

    def call_llm(self, params, **kwargs):
        print("hoi2")
        conversation = params["page_LLM"]["chat"]

        if not conversation:
            return None

        # Extract latest user input
        user_input = conversation.get_messages()
        if not user_input:
            return None

        print("hoi")
        # --- Run extraction chain ---
        try:
            result = chain.invoke({
                "input": user_input,
                "format_instructions": parser.get_format_instructions()
            })
        except Exception as e:
            return vkt.ChatResult(conversation, f"Error extracting parameters: {e}")

        print(type(result))
        
        # Save result in database
        input_dict = open_database()
        input_dict.update(result)
        save_database(input_dict)

        # --- Generate summary ---
        try:
            summary = summary_chain.invoke({"dict": json.dumps(result)})
            summary_text = summary.content
        except Exception as e:
            summary_text = f"Error generating summary: {e}"

        # Append summary to chat
        return vkt.ChatResult(conversation, f"Parameters extracted:\n{json.dumps(result, indent=2)}\n\nSummary:\n{summary_text}")
    
    
    def call_llm2(self, params, **kwargs):
        conversation = params["page_LLM"]["chat"]
        
        if not conversation:
            return None
        
        vkt_messages = conversation.get_messages()
        
        openai_messages = [
            {
                "role": "system",
                "content": "Always give a random number between 1 and 10 as output",
            }
        ]
        
        openai_messages.extend(vkt_messages)
        
        completion = client.chat.completions.create(
            model="gpt-4.1", messages=openai_messages
        )
        
        print(completion.choices)
        
        input_dict = open_database()
        input_dict["chargers"] = int(completion.choices[0].message.content)
        save_database(input_dict)
                
        return vkt.ChatResult(params["page_LLM"]["chat"], completion.choices[0].message.content)
    

    @staticmethod
    @PlotlyView('Aggregated Load Profile', duration_guess=10)
    def get_plotly_view_2(self, params, **kwargs):

        # ---- User Inputs ---- #
        input_dict = open_database()

        # ---- Time Vector ---- #
        times = pd.date_range("00:00", "23:45", freq="15min").strftime('%H:%M').tolist()
        n = len(times)

        # ---- EV Charging Profile ---- #
        charging_profile = np.zeros(n)
        start_idx = times.index(input_dict["charging_start"])
        for i in range(8):  # Charging over 2 hours
            idx = (start_idx + i) % n
            charging_profile[idx] = 1
        charging_profile = charging_profile * 11 * input_dict["chargers"]

        # ---- Provided Profiles ---- #
        # Household
        household_profile = [0.2,0.15,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
            0.1,0.1,0.1,0.1,0.1,0.15,0.2,0.25,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,
            0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.3,0.3,0.3,0.3,0.3,
            0.35,0.4,0.45,0.5,0.45,0.4,0.35,0.3,0.3,0.3,0.3,0.3,0.35,0.4,0.45,0.5,
            0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.7,
            0.7,0.7,0.7,0.7,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.2]

        # Commercial
        commercial_profile = [0.2,0.15,0.1,0.11,0.12,0.1,0.11,0.12,0.12,0.1,0.11,0.1,0.1,0.1,0.08,0.12,
            0.15,0.2,0.21,0.24,0.26,0.28,0.3,0.35,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,
            0.8,0.8,0.85,0.85,0.9,0.85,0.875,0.875,0.85,0.875,0.82,0.81,0.865,0.875,0.865,0.865,
            0.93,0.91,0.9,0.88,0.87,0.9,0.88,0.83,0.85,0.84,0.83,0.82,0.83,0.85,0.86,0.87,
            0.85,0.9,0.85,0.85,0.84,0.85,0.82,0.81,0.8,0.78,0.75,0.75,0.75,0.75,0.77,0.7,
            0.68,0.65,0.73,0.7,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.2]

        # Industrial
        industrial_profile = [0.1,0.11,0.1,0.09,0.1,0.09,0.1,0.11,0.1,0.09,0.1,0.1,0.1,0.11,0.1,0.1,
            0.1,0.1,0.1,0.11,0.1,0.1,0.2,0.32,0.41,0.52,0.63,0.7,0.81,0.92,0.98,0.99,
            0.98,0.99,1.0,0.97,0.98,0.99,1.0,1.0,0.99,0.98,1.0,0.98,1.0,1.0,0.98,1.0,
            0.99,0.98,0.99,0.97,0.98,0.99,1.0,0.99,0.98,0.97,0.99,0.98,0.99,0.98,0.99,1.0,
            1.0,0.99,0.98,1.0,0.99,0.88,0.8,0.71,0.62,0.53,0.44,0.31,0.22,0.21,0.2,0.22,
            0.2,0.21,0.22,0.2,0.2,0.21,0.2,0.2,0.2,0.21,0.2,0.19,0.2,0.21,0.2,0.19]

        # Solar
        solar_profile = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0.1,0.12,0.15,0.2,0.25,0.3,0.45,0.5,0.55,0.58,0.6,0.7,0.67,0.65,0.7,0.75,
            0.75,0.8,0.85,0.7,0.8,0.9,0.91,0.92,0.99,0.98,0.99,0.97,0.98,0.99,1,0.95,
            1,0.9,0.95,1,0.99,0.95,0.9,0.85,0.8,0.7,0.6,0.55,0.5,0.45,0.3,0.2,
            0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        # ---- Scale Profiles ---- #
        household_load = np.array(household_profile) * input_dict["households"] * 2
        commercial_load = np.array(commercial_profile) * input_dict["commercials"] * 5
        industrial_load = np.array(industrial_profile) * input_dict["industrials"] * 10
        solar_generation = np.array(solar_profile) * input_dict["solar_capacity_kw"]

        # ---- Plot ---- #
        fig = go.Figure()
        fig.add_bar(x=times, y=industrial_load, name="Industrial", marker_color="rgba(137, 207, 240, 1)")
        fig.add_bar(x=times, y=commercial_load, name="Commercial", marker_color="rgba(0, 0, 139, 1)")
        fig.add_bar(x=times, y=household_load, name="Household", marker_color="rgba(4, 118, 208, 1)")
        fig.add_bar(x=times, y=charging_profile, name="EV Charging", marker_color="rgba(148, 103, 189, 1)")
        fig.add_bar(x=times, y=-solar_generation, name="Solar (negative)", marker_color="rgba(255, 215, 0, 1)")

        fig.update_layout(
            title="Aggregated Load Profile (Stacked)",
            barmode='stack',
            xaxis_title="Time of Day",
            yaxis_title="Load (kW)",
            template='plotly_white',
            legend_title="Source"
        )


        fig = fig.to_json()

        return PlotlyResult(fig)

