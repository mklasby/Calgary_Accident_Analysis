from folium.plugins import heat_map
import pandas as pd
import numpy as np
import geojson
from model import Model
import folium
from folium.plugins import HeatMap


class Controller:
    def get_map(self):
        return self.mapa

    def load_data(self):
        print('Loading Data...')
        self.mdl = Model()
        print('...Data Loaded.')

    def display_data(self):
        for keyword, df in self.mdl.dfs.items():
            print(keyword, '\n', df.head())

    def get_frame(self, df_name):
        if df_name == 'temporal':
            return self.mdl.temporal_df
        return self.mdl.dfs[df_name]

    def add_geo_cols(self):
        df_names = self.mdl.dfs.keys()
        geo_cols = ['multiline', 'multilinestring',
                    'location', None, 'Point', 'POINT', 'cell_bounds', None]
        flip_list = [True, True, False, True, True, True, False, False]
        for name, col, flip in list(zip(df_names, geo_cols, flip_list)):
            print(
                f'Adding geometry column to {name} from {col}. Flip coords? {flip}')
            self.mdl.add_geo_col(name, col, flip)

    def add_cell_col(self):
        df_names = self.mdl.dfs.keys()
        for name in df_names:
            if name == "cells":
                continue
            print(f'Adding cell column to {name}')
            self.mdl.add_cell_col(name)

    def generate_map(self):
        width, height = 960, 600
        ne, sw = self.mdl.get_yyc_bounds()
        self.mapa = folium.Map(location=ne, width=width,
                               height=height, toFront=True)

        for cell in self.get_frame('cells')['cells']:
            cell.add_to(self.mapa)

        rect = folium.Rectangle(bounds=[ne, sw], weight=2, dash_array=(
            "4"), color='red', tooltip='Analysis Boundary').add_to(self.mapa)

        self.mapa.save('index.html')

    def get_cell_data():
        return

    def gen_heatmap(self):
        df = self.get_frame('volumes')
        data = []  # lat, lng, weight
        # TODO: https://github.com/python-visualization/folium/issues/1271

        for _, row in df.iterrows():
            volume = row['VOLUME']
            geometry = row['geometry']['coordinates']
            for points in geometry:
                lat = float(geometry[0])
                lon = float(geometry[1])
                data.append([lat, lon])
                heat_map = HeatMap(data, name="Volume")
                heat_map.add_to(self.mapa)
