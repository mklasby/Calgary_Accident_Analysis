'''
Controller
The interface between the jupyter notebook View and data Model. Parses input from View
and manipulates the model as required. Plotting functionality is implemented in this class.   
'''

from folium.plugins import heat_map
import pandas as pd
import numpy as np
import geojson
from model import Model
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import matplotlib.pyplot as plt
import seaborn as sns


class Controller:
    '''
    Interface between view and model 
    '''

    def load_data(self):
        print('Loading Data...')
        self.mdl = Model()
        print('...Data Loaded.')

    def display_data(self):
        for keyword, df in self.mdl.dfs.items():
            print(keyword, '\n', df.head())

    def get_frame(self, df_name):
        '''
        Queries model for a df by df_name
        '''
        if df_name == 'daily':
            return self.mdl.daily_df
        if df_name == 'hourly':
            return self.mdl.hourly_df
        return self.mdl.dfs[df_name]

    def add_geo_cols(self):
        '''
        Add GEOJSON object to each dataframe with Coordinate data under new column 'geometry'
        '''
        df_names = self.mdl.dfs.keys()
        geo_cols = ['multiline', 'multilinestring',
                    'location', None, 'Point', 'POINT', 'cell_bounds', None]
        flip_list = [True, True, False, True, True, True, False, False]
        for name, col, flip in list(zip(df_names, geo_cols, flip_list)):
            print(
                f'Adding geometry column to {name} from {col}. Flip coords? {flip}')
            self.mdl.add_geo_col(name, col, flip)

    def add_cell_col(self):
        '''
        Adds cell column to each dataframe with coordinate data. Each row will be placed 
        into a cell, or cells for a multiline string, depending on Coordinate data in that row.
        '''
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
        '''
        Generates 3 folium map objects with city bounds and cells overlaid on each.
        Maps stored as instance varible. 
        '''
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
        '''
        Adds static data to cells df for each static (time independent) df 
        '''
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
        self.cell_analysis()

    def cell_analysis(self):
        '''
        Add bins to cell df and calculates incidents per million trips
        '''
        print('Analyzing cell data...')
        cells = self.get_frame('cells')
        # TODO: Customize bin sizes if time permits
        # NOTE: https://stackoverflow.com/questions/32552027/with-pandas-cut-how-do-i-get-integer-bins-and-avoid-getting-a-negative-lowe
        # low, high = cells['volume_sum'].min(), cells['volume_sum'].max()
        # n_bins = 10
        # bin_edges = range(low, high, (high-low)/n_bins)
        # labels = ['(%d, %d]' %(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]

        cells['signal_bins'] = pd.cut(cells['signal_count'], bins=10)
        cells['sign_bins'] = pd.cut(cells['sign_count'], bins=10)
        cells['speed_bins'] = pd.cut(cells['avg_speed'], bins=10)
        cells['volume_bins'] = pd.cut(cells['volume_sum'], bins=10)
        cells['inc_per_mil_vol'] = \
            cells['incident_count'] / (cells['volume_sum']/1000000)
        self.get_cell_coords()

        print('...cells analyzed.')

    def count_points(self, cell_idx, df_name, col_name="cell"):
        '''
        Counts coord pairs within a cell and returns count
        :param: cell_idx: the index of the cell to check for points within
        :param: df_name the dataframe we are contining cells in 
        :param: col_name the column where we can find the cell location for each row

        '''
        df = self.get_frame(df_name)
        # print(f'testing cell {cell_idx}')
        counter = 0
        for _, cell in df[col_name].items():
            if cell_idx == cell:
                counter += 1
        return counter

    def get_cell_vol(self, cell_idx):
        '''
        Returns the sum of volumes within a cell at cell_idx. 
        NOTE: Volumes are NOT normalized by point count, we are interested in the
        total number of vehicles within the cell. 
        '''
        df = self.get_frame('volumes')
        volume_sum = 0
        num_points = 0
        for idx, row in df[['cell', 'VOLUME']].iterrows():
            cell_dict = row['cell']
            volume = row['VOLUME']
            if cell_idx in cell_dict:
                print(f'{cell_idx} is in {cell_dict} with volume {volume}')
                these_points = cell_dict[cell_idx]
                print(f'these points = {these_points}')
                volume_sum += volume
                num_points += these_points
        if num_points == 0:
            # if we have no data points, we cannot assume zero volume.
            # Therefore, fill with NaN to including in futher analysis.
            return np.nan
            # return 0
        print(volume_sum)
        return round(volume_sum)

    def get_avg_speed(self, cell_idx):
        '''
        Populates cells df with average speed per column. 
        NOTE: The speeds are normalized based on number of coordinate points for each
        road that we have data for. For eg., if we have 5 points at 50kph and 1 point at
        100kph, the average cell speed is (50*5 + 100*1)/(5+1) = 58.33kph
        '''
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
        return round(speed_sum/num_points, 2)  # round to 2 decimals

    def get_map(self, s):
        if s == "cell_map":
            return self.cell_map
        elif s == "speed_map":
            return self.speed_map
        elif s == "volume_map":
            return self.volume_map

    def get_polylines(self, geom, data, meta_data):
        '''
        Gets point cloud of speed/volume data in form of [ [lat, lon], speed/volume, tooltip name ]
        for plotting to maps. 
        :param: geom: this road segment's geometry
        :param: speed: this road's data
        :param: street_name: name of street to display in tooltip
        :returns: point cloud

        '''
        point_cloud = []
        tooltip = f'{meta_data}: {data}'
        for lines in geom['coordinates']:
            for points in lines:
                lat = float(points[0])
                lon = float(points[1])
                point_cloud.append([[lat, lon], data, tooltip])
        return point_cloud

    def draw_speed_map(self):
        '''get speed map, save to file and overwrite self.speed_map
        '''
        mapa = self.get_map('speed_map')
        df = self.get_frame('speeds')
        color_map = cm.LinearColormap(
            colors=['yellow', 'red'], vmin=df['SPEED'].min(), vmax=df['SPEED'].max())

        df['speed_lines'] = df.apply(lambda row: self.get_polylines(
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

    def add_volume_polylines(self):
        '''add volume polylines to volume heatmap and overwrite self.volume_map
        '''
        mapa = self.get_map('volume_map')
        df = self.get_frame('volumes')
        color_map = cm.LinearColormap(
            colors=['green', 'red'], vmin=df['VOLUME'].min(), vmax=df['VOLUME'].max())

        df['volume_lines'] = df.apply(lambda row: self.get_polylines(
            row['geometry'], row['VOLUME'], row['SECNAME']), axis=1)

        df = df.sort_values(by='VOLUME', ascending=False)

        for points in df['volume_lines'].tolist():
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

    def get_heat_points(self, geom, volume, vol_normalization):
        '''
        Gets heat map point cloud from df row
        :param: geom = geojson object (point or multilinestring)
        :param: volume = traffic volume of road
        :param: vol_normalization: metric to normalize vol with
        :return: point cloud data in form [lat, lon, volume_normalized]
        '''
        point_cloud = []
        vol_norm = volume/vol_normalization
        for lines in geom['coordinates']:
            for points in lines:
                lat = float(points[0])
                lon = float(points[1])
                point_cloud.append([lat, lon, vol_norm])
        return point_cloud

    def gen_heatmap(self):
        '''
        Generate volume heatmap based on data in volumes df. 
        Volumes normalized by volume median
        NOTE: Requires unreleased version 0.12 of folium, see below: 
        https://github.com/python-visualization/folium/issues/1271
        '''
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
        self.add_volume_polylines()
        mapa.save('volume_map.html')
        print("map saved")

    def melt_freeze(self, temps, freeze_temp, target=8):
        '''
        return pd.Series bool filter of  hours from 2018 such that Ti is an hour where the temperature fell below freeze_temp (C). Targets Ti are the ith hours after the freeze_temp occured. 

        :Params:    temps: pd.Series of hourly temperatures degrees C
                    freeze_temp: temperature of interest 
                    target: number of temperatures to capture after the freeze_temp. 
        :Returns:  pd.Series bool filter
        '''
        then = 0
        now = 1
        freeze = freeze_temp

        # list of target hours
        hours = []

        # pivot + 8 hours
        target = target

        #  Assume typical dangerous freeze near midnight and we wish to capture the morning     rush hour between 6-8am .

        while then < temps.size:
            target_hours = []
            try:
                if temps[now] <= freeze and temps[then] > freeze:
                    # print('hit')
                    idx = 0
                    while idx < target:
                        target_hours.append(now+idx)
                        idx += 1
                    hours.append(target_hours)
                then += 1
                now += 1
            except:
                then += 1
                now += 1
                continue

        mask_indices = []
        for targets in hours:
            for target in targets:
                if target in mask_indices:
                    continue
                else:
                    mask_indices.append(target)

        mask = []
        idx = 0
        while idx < temps.size:
            if idx in mask_indices:
                mask.append(True)
            else:
                mask.append(False)
            idx += 1
        mask = pd.Series(mask)
        print(f'There were {len(hours)} melt-freeze cycles in 2018!')
        return(mask)

    def get_super_plot(self, df, target_text, target_col, responding_col, x_label, y_label, title, binned=False, bin_col=None):
        '''
        Convenience function for creating plt.subplots from parameter inputs. 
        :params:
                    df: dataframe to plot from
                    target_text: string for titles and axis labels
                    target_col: column title from df to plot to x axis
                    responding_col: column title from df to plot to y axis
                    x_label: label for x axis on most plots
                    y_label: label for y axis on most plots
                    title: title of sub plots
                    binned: if true, will plot binned data for point and box plots
                    bin_col: column name where binned data stored
        :return:    plt.fig object

        '''
        fig, ((dist_ax, box_ax), (point_ax, line_ax)) = plt.subplots(
            nrows=2, ncols=2, figsize=(18, 12), )

        sns.distplot(df[target_col], kde=False, ax=dist_ax)
        dist_ax.set_xlabel(f'Count of {target_text}')
        dist_ax.set_ylabel(f'Frequency of {target_text} by Cell')
        dist_ax.set_title(f'Frequency of {target_text} Count by Cell')

        if binned:
            box_point_x = bin_col
        else:
            box_point_x = target_col

        sns.boxplot(x=box_point_x, y=responding_col, data=df, ax=box_ax)
        box_ax.set_xlabel(x_label)
        box_ax.set_ylabel(y_label)
        box_ax.set_title(title)

        sns.pointplot(x=box_point_x, y=responding_col,
                      hue=None, data=df, ax=point_ax)
        point_ax.set_xlabel(x_label)
        point_ax.set_ylabel(y_label)
        point_ax.set_title(title)

        if binned:
            box_ax.set_xticklabels(box_ax.get_xticklabels(),
                                   rotation=40, ha='right')
            point_ax.set_xticklabels(
                point_ax.get_xticklabels(), rotation=40, ha='right')

        sns.lineplot(x=target_col, y=responding_col, hue=None,
                     data=df, ax=line_ax, err_style=None)
        line_ax.set_xlabel(x_label)
        line_ax.set_ylabel(y_label)
        line_ax.set_title(title)

        fig.suptitle(f'Incidents vs. {target_text}', size='xx-large')
        fig.tight_layout(pad=5)
        fig.show()
        if responding_col == 'inc_per_mil_vol':
            plt.savefig(f'./plots/Incidents vs {target_text}_normalised.png')
        else:
            plt.savefig(f'./plots/Incidents vs {target_text}.png')
        return fig

    def get_cell_coords(self):
        '''
        Adds x, y coords as colums in cells df for each cell from cell index
        '''
        cells = self.get_frame('cells')
        cells_idx = list(cells.index)
        x_coords = []
        y_coords = []
        for idx in cells_idx:
            y_coords.append(idx // 10)
            x_coords.append(idx % 10)
        cells['x_coord'], cells['y_coord'] = x_coords, y_coords

    def cell_heatmap(self, df_name, col_name, ax, title):
        '''
        Gets seaborn heatmap plot axis for cell wise observations
        :params:
                df_name: name of df to plot
                col_name: name of column to plot data from
                ax: ax to plot data to
                title: title for ax
        :mutates: ax provided 
        '''
        df = self.get_frame(df_name)
        geo = df.groupby(['y_coord', 'x_coord'])[col_name].sum().unstack()
        ax = sns.heatmap(geo, annot=False, yticklabels=True, ax=ax)
        ax.set_title(title)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
