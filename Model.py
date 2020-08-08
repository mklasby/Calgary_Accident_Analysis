# model.py

import pandas as pd
import numpy as np
import geojson


class Model:
    '''
    A collection of pd.Dataframe objects 
    '''

    def __init__(self):
        self.dfs = {}
        incidents_df = pd.read_csv('Traffic_Incidents.csv')
        speeds_df = pd.read_csv('Speed_Limits.csv')
        cameras_df = pd.read_csv('Traffic_Camera_Locations.csv')
        signals_df = pd.read_csv('Traffic_Signals.csv')
        signs_df = pd.read_csv('Traffic_Signs.csv')
        volumes_df = pd.read_csv('Traffic_Volumes_for_2018.csv')
        self.dfs = {'incidents': incidents_df,
                    'speeds': speeds_df,
                    'cameras': cameras_df,
                    'signals': signals_df,
                    'signs': signs_df,
                    'volumes': volumes_df}

    def clean_coords(self):
        self.clean_points(self.dfs['incidents'], flipped=True)

    def clean_point(self, flipped=True):

    def clean_multiline(self, flipped=True):

    def get_coords(s: str):
    '''
    get coordinates from string data and convert to geojson multiline string object 
    '''
    cleaned = list(map(float, re.findall(r'[\-\d\.?]+', s)))
    print(cleaned)
    cleaned = flip_coords(cleaned)
    return MultiLineString(cleaned)

    def flip_coords(coords):
    flipped = []
    lats = []
    lons = []
    i = 0
    while i < len(coords):
        if i % 2 == 0:
            lons.append(coords[i])
        else:
            lats.append(coords[i])
        i += 1
    flipped = list(zip(lats, lons))
    return MultiLineString(flipped)
