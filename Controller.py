import pandas as pd
import numpy as np
import geojson


class Controller{


    my_vol = volumes_df['multilinestring']
    my_vol = my_vol.iloc[0]


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

    get_coords(my_vol)

    # volumes_df['multilinestring'] = volumes_df['multilinestring'].apply(lambda x: flip_coords(get_coords(x)))
    # print(volumes_df)

    # volumes_df['multilinestring'] = volumes_df['multilinestring'].apply(lambda x: flip_coords(get_coords(x)))

    # distance = 0
    # x0 = None
    # y0 = None
    # x1 = None
    # y1 = None
    # for latt, lonn, in coords:
    #     if (x0 == None):
    #         x0 = latt
    #         y0 = lonn
    #         continue
    #     x1 = latt
    #     y1 = lonn
    #     # print(x0, x1, y0, y1)
    #     distance += math.sqrt( (x1-x0)**2 + (y1-y0)**2 )
    #     x0 = latt
    #     y0 = lonn


    # print(distance)


}
