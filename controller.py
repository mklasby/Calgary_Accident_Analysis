from folium.plugins import heat_map
import pandas as pd
import numpy as np
import geojson
from model import Model
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import math


class Controller:

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

    def add_cell_object(self, bounds, tooltips):
        '''
        returns folium.rectangle with bounds = bounds and tooltip(s) = tooltips
        :param: bounds = cell_bounds 4 lat lon coords [ [lat, lon] [lat, lon] [lat, lon] [lat, lon] ]
        :param: tooltip: dict of k,v pairs where key = context and v = tooltip value
        ie., {'Average Speed in Cell': row['avg_speed']}

        populates cell_df with folium rectangle objects
        '''

        tips = ''

        for k, v in tooltips.items():
            tips += f'{k}: {v}\n'
        return folium.Rectangle(bounds=bounds, tooltip=tips)

    def generate_maps(self):
        print('Generating maps...')
        width, height = 960, 600
        ne, sw = self.mdl.get_yyc_bounds()

        cells = self.get_frame('cells')

        cells['cells'] = cells.apply(
            lambda row: self.add_cell_object(row['cell_bounds'],
                                             {'Cell': row.name,
                                              'Average Speed': row['avg_speed'],
                                              'Total Volume': row['volume_sum'],
                                              'Total Incidents': row['incident_count'],
                                              'Sign count': row['sign_count'],
                                              'Signal count': row['signal_count']
                                              }), axis=1)
        cells['vol_cells'] = cells.apply(lambda row: self.add_cell_object(
            row['cell_bounds'], {'Cell': row.name, 'Total Volume': row['volume_sum']}), axis=1)

        cells['speed_cells'] = cells.apply(lambda row: self.add_cell_object(
            row['cell_bounds'], {'Cell': row.name, 'Average Speed': row['avg_speed']}), axis=1)

        # NOTE: Deep copy of maps is not possible
        # https://github.com/python-visualization/folium/issues/1207

        # NOTE: We must duplicate the folium objects to add to each map:
        # https://stackoverflow.com/questions/25293579/multiple-leaflet-maps-on-same-page-with-same-options

        self.cell_map = folium.Map(location=[51.0478, -114.0593], width=width,
                                   height=height, toFront=True)
        self.speed_map = folium.Map(location=[51.0478, -114.0593], width=width,
                                    height=height, toFront=True)
        self.volume_map = folium.Map(location=[51.0478, -114.0593], width=width,
                                     height=height, toFront=True)

        for cell in self.get_frame('cells')['cells']:
            cell.add_to(self.cell_map)

        for cell in self.get_frame('cells')['vol_cells']:
            cell.add_to(self.volume_map)

        for cell in self.get_frame('cells')['speed_cells']:
            cell.add_to(self.speed_map)

        rect1 = folium.Rectangle(bounds=[ne, sw], weight=2, dash_array=(
            "4"), color='red', tooltip='Analysis Boundary')

        rect2 = folium.Rectangle(bounds=[ne, sw], weight=2, dash_array=(
            "4"), color='red', tooltip='Analysis Boundary')

        rect3 = folium.Rectangle(bounds=[ne, sw], weight=2, dash_array=(
            "4"), color='red', tooltip='Analysis Boundary')

        rect1.add_to(self.cell_map)
        rect2.add_to(self.speed_map)
        rect3.add_to(self.volume_map)

        self.cell_map.save('cell_map.html')
        self.speed_map.save('speed_map.html')
        self.volume_map.save('volume_map.html')

        # Drop extra folium vector object series once we are done adding to maps
        cells = self.get_frame('cells')
        cells = cells.drop(['vol_cells', 'speed_cells'], axis='columns')

        print('...maps generated.')

    def get_cell_data(self):
        print('Generating cell data...')
        cells = self.get_frame('cells')

        i = 0
        avg_speed = []
        volume_sum = []
        inc_count = []
        signal_count = []
        sign_count = []
        camera_count = []

        while i < 100:
            avg_speed.append(self.get_avg_speed(i))
            volume_sum.append(self.get_cell_vol(i))
            inc_count.append(self.count_points(i, 'incidents', 'cell'))
            signal_count.append(self.count_points(i, 'signals', 'cell'))
            sign_count.append(self.count_points(i, 'signs', 'cell'))
            camera_count.append(self.count_points(i, 'cameras', 'cell'))
            i += 1

        cells['avg_speed'] = avg_speed
        cells['volume_sum'] = volume_sum
        cells['incident_count'] = inc_count
        cells['sign_count'] = sign_count
        cells['signal_count'] = signal_count
        cells['camera_count'] = camera_count
        print('...cell data generated.')

    def count_points(self, cell_idx, df_name, col_name="cell"):
        df = self.get_frame(df_name)
        # print(f'testing cell {cell_idx}')
        counter = 0
        for _, cell in df[col_name].items():
            if cell_idx == cell:
                counter += 1
        return counter

    def get_cell_vol(self, cell_idx):
        df = self.get_frame('volumes')
        volume_sum = 0
        num_points = 0
        for idx, row in df[['cell', 'VOLUME']].iterrows():
            cell_dict = row['cell']
            volume = row['VOLUME']
            if cell_idx in cell_dict:
                # print(f'{cell_idx} is in {cell_dict}')
                these_points = cell_dict[cell_idx]
                # print(f'these points = {these_points}')
                volume_sum += volume
                num_points += these_points
        if num_points == 0:
            return np.nan
            # return 0
        return volume_sum

    def get_avg_speed(self, cell_idx):
        speeds = self.get_frame('speeds')
        speed_sum = 0
        num_points = 0
        for idx, row in speeds[['cell', 'SPEED']].iterrows():
            cell_dict = row['cell']
            speed = row['SPEED']
            if cell_idx in cell_dict:
                # print(f'{cell_idx} is in {cell_dict}')
                these_points = cell_dict[cell_idx]
                # print(f'these points = {these_points}')
                speed_sum += speed*these_points
                num_points += these_points
        if num_points == 0:
            return np.nan
            # return 0
        return round(speed_sum/num_points, 2)

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
                heat_map.add_to(self.volume_map)

    def get_map(self, s):
        if s == "cell_map":
            return self.cell_map
        elif s == "speed_map":
            return self.speed_map
        elif s == "volume_map":
            return self.volume_map

    def get_speed_list(self, geom, speed, street_name):
        # print(geom)
        point_cloud = []
        tooltip = f'{street_name} speed limit: {speed}'
        for lines in geom['coordinates']:
            for points in lines:
                lat = float(points[0])
                lon = float(points[1])
                point_cloud.append([[lat, lon], speed, tooltip])
        return point_cloud

    def draw_speed_map(self):
        '''get speed map
        @args: speed_frame :pd.DataFrame to map
        @return folium map NOTE: front end will simply render map.html from assets
        '''
        mapa = self.get_map('speed_map')
        df = self.get_frame('speeds')
        color_map = cm.LinearColormap(
            colors=['yellow', 'red'], vmin=df['SPEED'].min(), vmax=df['SPEED'].max())

        df['speed_lines'] = df.apply(lambda row: self.get_speed_list(
            row['geometry'], row['SPEED'], row['STREET_NAME']), axis=1)

        df = df.sort_values(by='SPEED', ascending=False)

        for points in df['speed_lines'].tolist():
            locations = []
            colors = []
            tooltip = points[0][2]
            for point in points:
                loc = point[0]
                color = point[1]
                tooltip = point[2]
                locations.append(loc)
                colors.append(color)
            this_line = folium.PolyLine(
                locations=locations, tooltip=tooltip, color=color_map(color))
            this_line.add_to(mapa)
        mapa.save('speed_map.html')
        print("map saved")

    def get_heat_points(self, geom, volume, vol_normalization):
        point_cloud = []
        vol_norm = volume/vol_normalization
        for lines in geom['coordinates']:
            for points in lines:
                lat = float(points[0])
                lon = float(points[1])
                point_cloud.append([lat, lon, vol_norm])
        return point_cloud

    def gen_heatmap(self):
        mapa = self.get_map('volume_map')
        df = self.get_frame('volumes')
        data = []  # lat, lng, weight

        vol_median = df['VOLUME'].median()
        df['heat_map_points'] = df.apply(lambda row: self.get_heat_points(
            row['geometry'], row['VOLUME'], vol_median), axis=1)
        data = []
        for points in df['heat_map_points'].tolist():
            for point in points:
                data.append(point)
        heat_map = HeatMap(data, radius=10, blur=15)
        heat_map.add_to(mapa)
        mapa.save('volume_map.html')
        print("map saved")
