'''
Model
The data layer of our project. This class stores the various dataframes used in our
analysis and manipulates data views at request of controller class. 
'''

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
        incidents_df = pd.read_csv('./data/Traffic_Incidents.csv')
        incidents_df = self.get_2018_inc(incidents_df)
        speeds_df = pd.read_csv('./data/Speed_Limits.csv')
        cameras_df = pd.read_csv('./data/Traffic_Camera_Locations.csv')
        signals_df = pd.read_csv('./data/Traffic_Signals.csv')
        signs_df = pd.read_csv('./data/Traffic_Signs.csv')
        volumes_df = pd.read_csv('./data/Traffic_Volumes_for_2018.csv')
        cells_df = self.get_cells_df()

        # time specific data
        # TODO: uncommment below to get temporal data
        self.hourly_df, self.daily_df = self.get_temporal_data()

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
        '''
        Gets time dependent data and joins with incident data. 
        Adds pd.datetime column for grouping data
        Adds time dependent data bins for plotting
        Splits data into hourly and daily dataframes
        '''
        weather_df = pd.DataFrame()
        for i in range(1, 13):
            print(f'Getting weather at yyc for month {i} in 2018')
            month = self.get_weather(50430, 2018, i)
            weather_df = weather_df.append(month, ignore_index=True)
        # print(weather_df)
        weather_df['date'] = pd.to_datetime(weather_df['Date/Time (LST)'])

        incidents = pd.read_csv('./data/Traffic_Incidents.csv')
        incidents['date'] = pd.to_datetime(incidents['START_DT'])
        mask_2018 = incidents['date'].dt.year == 2018
        incidents = incidents[mask_2018]
        incidents = incidents.resample('H', on='date')['Count'].count()
        incidents.name = 'incidents'

        hourly_df = pd.merge(weather_df, incidents, on='date')
        daily_df = hourly_df.copy(deep=True)

        hourly_df['temp_bins'] = pd.cut(
            hourly_df['Temp (C)'], bins=[-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40])
        hourly_df['vis_bins'] = pd.cut(hourly_df['Visibility (km)'], bins=[
                                       0, 1, 2, 3, 5, 10, 20, 40, 80])

        daily_df = daily_df[['date', 'incidents', 'Visibility (km)', 'Temp (C)']].resample(
            'D', on='date').agg({'incidents': np.sum, 'Visibility (km)': np.mean, 'Temp (C)': np.mean})
        daily_df.rename(columns={'incidents': 'sum_daily_incidents',
                                 'Visibility (km)': 'avg_daily_vis', 'Temp (C)': 'avg_daily_temp'}, inplace=True)
        daily_df['temp_bins'] = pd.cut(
            daily_df['avg_daily_temp'], bins=[-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40])
        daily_df['vis_bins'] = pd.cut(daily_df['avg_daily_vis'], bins=[
                                      0, 1, 2, 3, 5, 10, 20, 40, 80])

        return hourly_df, daily_df

    def get_weather(self, station, year, month, daily=False):
        ''' 
        Gets climate data from climate.weather.gc.ca
        :params: 
                station: station to pull data from
                year: year to pull data from
                month: month to pull data from
                daily = sets timeframe parameter in url parameter
        :returns: requested weather data in dataframe

        '''
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
        '''
        Gets max/min lat and lon to plot city boundary
        '''
        yyc_map = pd.read_csv('./data/City_Boundary_layer.csv')
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
        Create cell dfs and adds to self.dfs
        Populates cell geometry in df
        '''
        ne, sw = self.get_yyc_bounds()

        # create 10 spaces with 11 cuts
        cols = np.linspace(sw[1], ne[1], num=11)
        rows = np.linspace(sw[0], ne[0], num=11)

        points = []
        for col in cols:
            for row in rows:
                points.append([row, col])
        cell_bounds = []
        cell_idx = 0
        for idx_y, row in enumerate(rows):
            if idx_y == 10:
                break
            for idx_x, col in enumerate(cols):
                if idx_x == 10:
                    break
                bottom_left = [rows[idx_y], cols[idx_x]]
                top_right = [rows[idx_y+1], cols[idx_x+1]]

                # cell bounds, sw and ne corners
                cell_bounds.append([bottom_left, top_right])
                cell_idx += 1

        cells_df = pd.DataFrame({'cell_bounds': cell_bounds})
        return cells_df

    def clean_geo_data(self, s: str, to="Point", flip=True):
        '''
        get coordinates from string data and convert to geojson object
        :params: 
                s: string to clean, expecting Point or multilinestring from csv
                to: geojson object to convert to
                flip: if true, flips lon lat pairs to lat lon
        :return: geojson object point or multilinestring
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
        '''
        Adds geometry column to dataframe which will contain a geojson representation of messy
        string coord data in csv
        :params:    df_name: name of df to add column to 
                    geo_col_name: name of column where string coord data is found
                    flip: boolean, will flip lon lat to lat lon
        '''
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
            # Polygon data not required.
            return
        else:  # temporal data not requred.
            return

    def add_cell_col(self, df_name):
        '''
        Applys place_in_cell lambda function to df named df_name 
        '''
        df = self.dfs[df_name]
        df['cell'] = df['geometry'].apply(lambda x: self.place_in_cell(x))

    def place_in_cell(self, geom):
        '''
        places geojson object within a cell (or cells for multiline)
        :param: geom: geojson Point or MultiLineString object
        :returns:
                    if geom isa Point: cell idx where point exists
                    if geom isa multiLineString: dict in form of {cell_idx: num_points in cell, cell_idx...}
        '''
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
            # NOTE: Polygon not used, return warning below if accidently passed to function
            return ('?POLYGON?')

    def flip_coords(self, coords):
        '''
        flips all elements of list such that [x,y,i,j] = [ [y,x], [j,i] ]
        :return: flipped list
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

    def get_2018_inc(self, df):
        '''
        Filters incident df for only dates in 2018.
        :param: dataframe to filer
        :returns: filtered df
        '''
        df['date'] = pd.to_datetime(df['START_DT'])
        mask_2018 = df['date'].dt.year == 2018
        df = df.loc[mask_2018]
        return df
