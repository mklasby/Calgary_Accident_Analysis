import pandas as pd
import numpy as np
import geojson
from model import Model
import folium


class Controller:

    def load_data(self):
        print('Loading Data...')
        self.mdl = Model()
        print('...Data Loaded.')

    def display_data(self):
        for keyword, df in self.mdl.dfs.items():
            print(keyword, '\n', df.head())

    def get_frame(self, df_name):
        return self.mdl.dfs[df_name]

    def add_geo_cols(self):
        df_names = self.mdl.dfs.keys()
        geo_cols = ['multiline', 'multilinestring',
                    'location', None, 'Point', 'POINT', 'cell_bounds']
        flip_list = [True, True, False, True, True, True, False]
        for name, col, flip in list(zip(df_names, geo_cols, flip_list)):
            print(
                f'Creating geometry column in df {name} from {col}. Am I flipping coords? {flip}')
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
        mapa = folium.Map(location=ne, width=width,
                          height=height, toFront=True)

        for cell in self.get_frame('cells')['cells']:
            cell.add_to(mapa)

        rect = folium.Rectangle(bounds=[ne, sw], weight=2, dash_array=(
            "4"), color='red', tooltip='Analysis Boundary').add_to(mapa)

        mapa.save('index.html')
