# model.py

from geojson import geometry
import numpy as np
from numpy.lib.polynomial import poly
import pandas as pd
from geojson import Point, MultiLineString, Polygon
import geopandas as gpd
import re
import folium
# import urllib as url
# import requests


class Model:
    '''
    A collection of pd.Dataframe objects
    '''

    def __init__(self):
        self.dfs = {}
        incidents_df = pd.read_csv('Traffic_Incidents.csv')
        incidents_df = self.get_2018_inc(incidents_df)
        speeds_df = pd.read_csv('Speed_Limits.csv')
        cameras_df = pd.read_csv('Traffic_Camera_Locations.csv')
        signals_df = pd.read_csv('Traffic_Signals.csv')
        signs_df = pd.read_csv('Traffic_Signs.csv')
        volumes_df = pd.read_csv('Traffic_Volumes_for_2018.csv')
        cells_df = self.get_cells_df()
        # time specific data
        # TODO: uncommment below to get temporal data
        # self.temporal_df = self.get_temporal_data()
        # static data
        self.dfs = {'speeds': speeds_df,  # line geometry
                    'volumes': volumes_df,  # line geometry
                    'incidents': incidents_df,  # point geometry
                    'cameras': cameras_df,  # point geometry
                    'signals': signals_df,  # point geometry
                    'signs': signs_df,  # point geometry
                    'cells': cells_df,  # polygon geometry
                    }

    def get_temporal_data(self):
        weather_df = pd.DataFrame()
        for i in range(1, 13):
            print(f'Getting weather at yyc for month {i} in 2018')
            month = self.get_weather(50430, 2018, i)
            weather_df = weather_df.append(month, ignore_index=True)
        weather_df['date'] = pd.to_datetime(weather_df['Date/Time'])

        incidents = pd.read_csv('Traffic_Incidents.csv')
        incidents['date'] = pd.to_datetime(incidents['START_DT'])
        mask_2018 = incidents['date'].dt.year == 2018
        incidents = incidents[mask_2018]
        incidents = incidents.resample('H', on='date')['Count'].count()
        incidents.name = 'incidents'

        temporal_df = pd.merge(weather_df, incidents, on='date')
        return temporal_df

    def get_weather(self, station, year, month, daily=False):
        """ returns a dataframe with weather data from climate.weather.gc.ca"""
        # TODO: Refactor
        if daily:
            timeframe = 2
        else:
            timeframe = 1

        url_template = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={station}&Year={year}&Month={month}&Day=14&timeframe={timeframe}&submit=Download+Data"
        url = url_template.format(
            station=station, year=year, month=month, timeframe=timeframe)
        weather_data = pd.read_csv(
            url)

        weather_data.columns = [col.replace(
            '\xb0', '') for col in weather_data.columns]
        return weather_data

    def get_yyc_bounds(self):
        yyc_map = pd.read_csv('City_Boundary_layer.csv')
        geom = yyc_map.the_geom[0]
        geom = MultiLineString(self.clean_geo_data(
            geom, 'MultiLineString', True))

        ne = [float('-inf'), float('-inf')]
        sw = [float('+inf'), float('+inf')]

        # find maximum  /minumum coords
        for lines in geom['coordinates']:
            for point in lines:
                if point[0] > ne[0]:
                    ne[0] = point[0]
                if point[1] > ne[1]:
                    ne[1] = point[1]
                if point[0] < sw[0]:
                    sw[0] = point[0]
                if point[1] < sw[1]:
                    sw[1] = point[1]
        return (ne, sw)

    def get_cells_df(self):
        '''
        Create cell dfs add to self.dfs, bounds geometry
        '''
        ne, sw = self.get_yyc_bounds()

        # create 10 spaces with 11 cuts
        cols = np.linspace(sw[1], ne[1], num=11)
        rows = np.linspace(sw[0], ne[0], num=11)

        points = []
        for col in cols:
            for row in rows:
                points.append([row, col])

        cells = []
        cell_bounds = []
        geometry = []
        cell_idx = 0
        for idx_y, row in enumerate(rows):
            if idx_y == 10:
                break
            for idx_x, col in enumerate(cols):
                if idx_x == 10:
                    break
                bottom_left = [rows[idx_y], cols[idx_x]]
                bottom_right = [rows[idx_y], cols[idx_x]+1]
                top_right = [rows[idx_y+1], cols[idx_x+1]]
                top_left = [rows[idx_y]+1, cols[idx_x]]

                # cell bounds, sw and ne corners
                cell_bounds.append([bottom_left, top_right])

                # TODO: Add geoJSON polygon class to geometry colum
                # geometry.append(Polygon([])

                # Store cells as folium rectangle objects, change tooltip here.
                # TODO: Move to controller class
                cells.append(folium.Rectangle(
                    bounds=[bottom_left, top_right], tooltip=f'Cell: {cell_idx}'))
                cell_idx += 1

        cells_df = pd.DataFrame({'cells': cells, 'cell_bounds': cell_bounds})

        return cells_df

    def clean_geo_data(self, s: str, to="Point", flip=True):
        '''
        get coordinates from string data and convert to geojson object
        '''
        if to == 'Point':
            cleaned = list(map(float, re.findall(r'[\-?\d\.?]+', s)))
            if flip:
                cleaned = self.flip_coords(cleaned)
                cleaned = [cleaned[0][0], cleaned[0][1]]
            return Point(cleaned)

        elif to == 'MultiLineString':
            cleaned = []
            lines = re.findall(r'\((.*?)\)', s)
            for line in lines:
                this_line = list(map(float, re.findall(r'[\-?\d\.?]+', line)))
            if flip:
                cleaned.append(self.flip_coords(this_line))
            else:
                cleaned.append(self.this_line)
            return MultiLineString(cleaned)
        else:
            return -1

    def add_geo_col(self, df_name, geo_col_name='Point', flip=True):
        df = self.dfs[df_name]
        point_col_names = ['Point', 'POINT', 'location']
        mls_col_names = ['multiline', 'multilinestring']
        poly_col_names = ['cell_bounds']
        if geo_col_name in point_col_names:
            df['geometry'] = df[geo_col_name].apply(
                lambda x: self.clean_geo_data(x, to="Point", flip=flip))
        elif geo_col_name in mls_col_names:
            df['geometry'] = df[geo_col_name].apply(
                lambda x: self.clean_geo_data(x, to="MultiLineString", flip=flip))
        elif geo_col_name == None:  # special case for cameras_df
            df['geometry'] = list(zip(df['latitude'], df['longitude']))
            df['geometry'] = df['geometry'].apply(lambda x: Point(x))
        elif geo_col_name in poly_col_names:
            # TODO: Deal with polygons
            return
        else:  # skip temporal data
            return

    def add_cell_col(self, df_name):
        df = self.dfs[df_name]
        df['cell'] = df['geometry'].apply(lambda x: self.place_in_cell(x))

    def place_in_cell(self, geom):
        if isinstance(geom, Point):
            lat = geom['coordinates'][0]
            lon = geom['coordinates'][1]
            for idx, cell in self.dfs['cells']['cell_bounds'].items():
                sw = cell[0]
                ne = cell[1]
                if (lat >= sw[0]) and (lat < ne[0]) and (lon >= sw[1]) and (lon < ne[1]):
                    # print(f'HIT! {lat}, {lon} is in cell {idx} with bounds {sw, ne}')
                    return idx
            return -1

        elif isinstance(geom, MultiLineString):
            cell_counts = {}
            for lines in geom['coordinates']:
                for point in lines:
                    lat = point[0]
                    lon = point[1]
                    for idx, cell in self.dfs['cells']['cell_bounds'].items():
                        sw = cell[0]
                        ne = cell[1]
                        if (lat >= sw[0]) and (lat < ne[0]) and (lon >= sw[1]) and (lon < ne[1]):
                            cell_counts[idx] = cell_counts.get(idx, 0)+1
            return cell_counts

        elif isinstance(geom, Polygon):
            # TODO: implement if neeeded
            return ('?POLYGON?')

    def flip_coords(self, coords):
        '''
        flips all elements of list such that [x,y,i,j] = [ [y,x], [j,i] ]
        '''
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
        return flipped

    def get_avg_speed(self, cell_idx):
        '''
        calculates average speed based on cell index 
        '''
        speed_sum = 0
        num_points = 0
        speeds = self.dfs['speeds']
        for _, row in speeds[['cell', 'SPEED']].iterrows():
            cell_dict = row['cell']
            speed = row['SPEED']
            if cell_idx in cell_dict:
                these_points = cell_dict[cell_idx]
                speed_sum += speed*these_points
                num_points += these_points
        if num_points == 0:
            return np.nan
        return speed_sum/num_points

    def count_incidents(self, cell_idx):
        incidents = self.dfs('incidents')
        counter = 0
        for _, cell in incidents['cell'].items():
            if cell_idx == cell:
                counter += 1
        return counter

    def get_2018_inc(self, df):
        df['date'] = pd.to_datetime(df['START_DT'])
        mask_2018 = df['date'].dt.year == 2018
        df = df.loc[mask_2018]
        return df
