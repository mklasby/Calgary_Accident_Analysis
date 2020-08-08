import pandas as pd
import numpy as np
import geojson


class geo_cleaner:
    def clean_multi_line(s: str, flip=True):
        '''
        get coordinates from string data and convert to geojson multiline string object 
        '''
        cleaned = list(map(float, re.findall(r'[\-?\d\.?]+', s)))
        if flip:
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
        return flipped

    def clean_point_coords(s: str, flip=True):
        cleaned = list(map(float, re.findall(r'[\-\d\.?]+', s)))
        # print(cleaned)
        if flip:
            cleaned = flip_coords(cleaned)
        return Point(cleaned)


display(signals_df['Point'].apply(lambda x: clean_point_coords(x)))
display(signs_df['POINT'].apply(lambda x: clean_point_coords(x)))
display(incidents_df['location'].apply(
    lambda x: clean_point_coords(x, flip=False)))
display(volumes_df['multilinestring'].apply(lambda x: clean_multi_line(x)))
display(speeds_df['multiline'].apply(lambda x: clean_multi_line(x)))
cameras_df['Point'] = list(
    zip(cameras_df['latitude'], cameras_df['longitude']))
cameras_df['Point'] = cameras_df['Point'].apply(lambda x: Point(x))
display(cameras_df)
