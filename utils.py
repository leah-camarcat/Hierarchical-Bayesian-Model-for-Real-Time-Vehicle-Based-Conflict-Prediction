import numpy as np
import pandas as pd
import os
#from os.path import exists
import glob
#import haversine
from haversine import haversine, Unit
import math
import geopandas as gpd
from shapely.geometry import Point, LineString
from convertbng.util import convert_bng, convert_lonlat

# Load the shapefile of the road network
# roads = gpd.read_file("Major_Road_Network_2018_Open_Roads/Major_Road_Network_2018_Open_Roads.shp")
ilds = pd.read_csv('ild_loc_ab.csv', decimal=',')
ilds = ilds.iloc[:, 2:]

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two latitude/longitude points.
    Bearing is the direction from point 1 to point 2 in degrees.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1

    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)

    initial_bearing = math.atan2(x, y)
    # Convert from radians to degrees and normalize to 0-360
    bearing = (math.degrees(initial_bearing) + 360) % 360
    return bearing

def find_motorway(lat, lon, roads):
    res = convert_bng(lon,lat)
    vehicle_point = Point(res)  # Longitude first for Point()
    
    # Find the closest road segment
    roads['distance'] = roads.geometry.distance(vehicle_point)
    nearest_road = roads.loc[roads['distance'].idxmin()]
    
    motorway_name = nearest_road['roa_number']  # Adjust this column name based on your shapefile
    return motorway_name

def find_motorway_and_direction(lat1, lon1, lat2, lon2, roads):
    # Calculate bearing
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    
    # Find nearest motorway for the starting point
    motorway_name = find_motorway(lat1, lon1, roads)
    # Interpret direction based on bearing
    if 45 <= bearing < 135:
        direction = "Eastbound"
    elif 135 <= bearing < 225:
        direction = "Southbound"
    elif 225 <= bearing < 315:
        direction = "Westbound"
    else:  # Covers 315 to 360 and 0 to 45
        direction = "Northbound"
    
    return motorway_name, direction

def ILD_finder(ilds, lat, long, dir):

    coords_1 = (lat, long)
    
    # Create a boolean mask for non-NaN values
    valid_mask = ~ilds.iloc[:, -2].isna() & ~ilds.iloc[:, -1].isna()

    # Filter valid ILDs
    valid_ilds = ilds[valid_mask].copy()
    # check for A and B
    valid_ilds = valid_ilds[valid_ilds['ILDs '].str.contains(dir)].copy()

    # Extract coordinates from valid ILDs
    coords_2 = valid_ilds.iloc[:, -2:].values
    # Calculate distances
    distances = np.array([haversine(coords_1, tuple(coord), unit=Unit.KILOMETERS) for coord in coords_2])
    
    # Find the index of the minimum distance
    ind = np.argmin(distances)
    
    road_name = valid_ilds.iloc[ind, 0]
    ild_name = valid_ilds.iloc[ind, 2]
    #print('check ilds', valid_ilds.iloc[ind, :])
    #print('road ild names', road_name, ild_name)

    return road_name, ild_name

def trafficDataFinder(ild_name, date, time, lookback, rootpath, dir, ild_order, motorway, lat1, lon1):
    filename = os.path.join(rootpath, ild_name + '.csv')
    target_road = ilds[ilds['ILDs '] == ild_name]['Road id '].values[0]
    ild_list = ilds[ilds['Road id '] == target_road]
    m_ilds = ilds[ilds['Road id '] == motorway]
    # file_list = np.array(m_ilds['ILDs '])  + '.csv'
    # Check if the file exists
    if os.path.exists(filename):
        ild_data = pd.read_csv(filename, delimiter='\t')
        df = ild_data[ild_data['Date'] == date]
        time_indices = df.index[df['Time'] == time].tolist()
        index = time_indices[0]
        # Get data with lookback
        start_index = max(0, index - lookback)
        data = df.loc[start_index:index]
        lanes = data['Number of Lanes'].values[0]
        flow = data.iloc[:,19:19 + lanes].sum(axis = 1)

    # add files sorted the first one as the true one
    elif (os.path.exists(filename)==False) or (flow!=None and np.sum(flow < 0) > 0):
        # List all csv files in the rootpath 
        files = glob.glob(os.path.join(rootpath, "*.csv"))
        file_list = []
        for file in files:
            if not ild_list[ild_list['ILDs '] == file[42:-4]].empty:
                if (file[-5] == dir):
                    file_list.append(file)

        # Find the closest ILD file
        closest_files = []
        files_with_diff = []
        closest_diff = float('inf')
        target_num = int(ild_name[:-1])
        target_letter = ild_name[-1]

        for file in file_list:
            #file_num = int(file[:4])
            #file_letter = file[-5]
            if file[-5].isnumeric():
                file_num = int(file[42:-6])
                file_letter = file[-6]
            else:
                file_num = int(file[42:-5])
                file_letter = file[-5]
            
            # incorporate the increase or decrease of numbers wanted:
            diff = abs(target_num - file_num)
            files_with_diff.append((diff, file))
            # if diff < 0 and ild_order == 'increase':
                    # files_with_diff.append((diff, file))
            # elif diff > 0 and ild_order == 'decrease':
                # files_with_diff.append((diff, file))
                
        files_with_diff.sort(key=lambda x: x[0])
        sorted_files = [file for _, file in files_with_diff]
        #print('sorted files:', sorted_files)

        if closest_files is None:
            raise FileNotFoundError(f"No file found matching ILD name {ild_name}")
        

        ild_data = pd.read_csv(sorted_files[0], delimiter='\t')
        k=0
        while ild_data.empty:
            k += 1
            if k == len(sorted_files):
                ild_data = pd.DataFrame(np.nan, index=[0, 1, 2, 3, 4], columns=ild_data.columns)
                print('breaking')
                break
            ild_data = pd.read_csv(sorted_files[k], delimiter='\t')

        if not ild_data.empty:
            # Filter data for the specific date
            df = ild_data[ild_data['Date'] == date]

            # Find the specific time index
            time_indices = df.index[df['Time'] == time].tolist()
            if not time_indices:
                raise ValueError(f"No data found for time {time} on date {date}")
                 
            else:
                index = time_indices[0]
                
                # Get data with lookback
                start_index = max(0, index - lookback)
                data = df.loc[start_index:index]

                # check the data is not missing with positive flow
                lanes = data['Number of Lanes'].values[0]
                flow = data.iloc[:, 19:19 + lanes].sum(axis = 1)
                l = 1
                if np.sum(flow < 0) > 0:
                    while np.sum(flow < 0) > 1:
                        if os.path.exists(sorted_files[l]):
                            ild_data = pd.read_csv(sorted_files[l], delimiter='\t')
                            df = ild_data[ild_data['Date'] == date]
                            time_indices = df.index[df['Time'] == time].tolist()
                            index = time_indices[0]
                            start_index = max(0, index - lookback)
                            lanes = data['Number of Lanes'].values[0]
                            data = df.loc[start_index:index]
                            flow = data.iloc[:,19:19 + lanes].sum(axis = 1)
                        #print('flow in loop', flow)
                        l += 1
    lat2 = m_ilds[m_ilds['ILDs '] == ild_data.iloc[0,0]]['Latitude'].values[0]
    lon2 = m_ilds[m_ilds['ILDs '] == ild_data.iloc[0,0]]['Longitude'].values[0]
    coords1 = (lat1,lon1)
    coords2 = (lat2,lon2)
    dist = haversine(coords1, coords2, unit=Unit.KILOMETERS)
    return data, dist


################ OLD #############################

def ILD_finder_oldold(ilds, lat, long):
    coords_1 = (lat, long)
    distance = []
    for i in range(len(ilds)):
        if ilds.iloc[i, -2] == np.nan or ilds.iloc[i, -1] == np.nan:
            distance.append(np.nan)
        else:
            coords_2 = (ilds.iloc[i, -2], ilds.iloc[i, -1])
            distance.append(hs.haversine(coords_1, coords_2))
    ind = np.nanargmin(distance)
    road_name = ilds.iloc[ind, 0]
    ild_name = ilds.iloc[ind, 2]
    print(road_name, ild_name)
    return(road_name, ild_name)

def ILD_finder_old(ilds, lat, long):
    coords_1 = (lat, long)
    
    # Create a boolean mask for non-NaN values
    valid_mask = ~ilds.iloc[:, -2].isna() & ~ilds.iloc[:, -1].isna()
    
    # Filter valid ILDs
    valid_ilds = ilds[valid_mask]
    
    # Extract coordinates from valid ILDs
    coords_2 = valid_ilds.iloc[:, -2:].values
    # Calculate distances
    distances = np.array([haversine(coords_1, tuple(coord), unit=Unit.KILOMETERS) for coord in coords_2])
    
    # Find the index of the minimum distance
    ind = np.argmin(distances)
    
    road_name = valid_ilds.iloc[ind, 0]
    ild_name = valid_ilds.iloc[ind, 2]
    print('check ilds', valid_ilds.iloc[ind, :])
    print('road ild names', road_name, ild_name)
    # distinguish A and B !!!!
    return road_name, ild_name


#ilds = pd.read_csv('ILDS_loc.csv', delimiter=';', decimal=',')
ilds = pd.read_csv('ild_loc_ab.csv', decimal=',')
ilds = ilds.iloc[:,2:]
ilds = ilds.dropna()
#print(ilds)
# ild_name = ILD_finder(ilds, 51.747535, -0.4064)

def trafficDataFinder_nope(ild_name, date, time, lookback, rootpath):
    filename = os.path.join(rootpath, ild_name + '.csv')
    target_road = ilds[ilds['ILDs '] == ild_name]['Road id '].values[0]
    ild_list = ilds[ilds['Road id '] == target_road]
    
    # Check if the file exists
    if os.path.exists(filename):
        ild_data = pd.read_csv(filename, delimiter='\t')
        df = ild_data[ild_data['Date'] == date]
        time_indices = df.index[df['Time'] == time].tolist()
        index = time_indices[0]
        # Get data with lookback
        start_index = max(0, index - lookback)
        data = df.loc[start_index:index]
    
    # add files sorted the first one as the true one !!!
    else:
        # List all csv files in the rootpath
        files = glob.glob(os.path.join(rootpath, "*.csv"))
        file_list = []
        for file in files:
            if not ild_list[ild_list['ILDs '] == file[42:-4]].empty:
                file_list.append(file)
        # Find the closest ILD file
        closest_files = []
        files_with_diff = []
        closest_diff = float('inf')
        target_num = int(ild_name[:-1])
        target_letter = ild_name[-1]

        for file in file_list:
            if file[-5].isnumeric():
                file_num = int(file[42:-6])
                file_letter = file[-6]
            else:
                file_num = int(file[42:-5])
                file_letter = file[-5]
            
            #if ilds[ilds['ILDs '] == ild_name]['Road id ']
            if target_road == 'M40':
                target_letter = 'A'
            if file_letter == target_letter:
                diff = abs(target_num - file_num)
                files_with_diff.append((diff, file))
                # if diff < closest_diff:
                #     closest_diff = diff
                #     closest_files.append((diff, file))
                    # closest_file = file
        files_with_diff.sort(key=lambda x: x[0])
        sorted_files = [file for _, file in files_with_diff]
        #closest_files = sorted(closest_files, key=lambda x: x[0])[:30]

        if closest_files is None:
            raise FileNotFoundError(f"No file found matching ILD name {ild_name}")
        
        ild_data = pd.read_csv(sorted_files[0], delimiter='\t')
        k=0
        while ild_data.empty:
            k += 1
            if k == len(sorted_files):
                ild_data = pd.DataFrame(np.nan, index=[0, 1, 2, 3, 4], columns=ild_data.columns)
                print('breaking')
                break
            ild_data = pd.read_csv(sorted_files[k], delimiter='\t')
        print('ild_data', ild_data)
        if not ild_data.empty:
            # Filter data for the specific date
            df = ild_data[ild_data['Date'] == date]

            # Find the specific time index
            time_indices = df.index[df['Time'] == time].tolist()
            # print('time_indices', time_indices)
            if not time_indices:
                # data = pd.DataFrame(np.nan, index=[0, 1, 2, 3, 4], columns=ild_data.columns)
                raise ValueError(f"No data found for time {time} on date {date}")
                # find closest existing time ! 
                #time_indices 
            else:
                index = time_indices[0]
                
                # Get data with lookback
                start_index = max(0, index - lookback)
                data = df.loc[start_index:index]

                # check the data is not missing with positive flow
                lanes = data['Number of Lanes'].values[0]
                flow = data.iloc[:, 19:19 + lanes].sum(axis = 1)
                l = 1
                if np.sum(flow < 0) > 0:
                    while np.sum(flow < 0) > 1:
                        ild_data = pd.read_csv(sorted_files[l], delimiter='\t')
                        df = ild_data[ild_data['Date'] == date]
                        time_indices = df.index[df['Time'] == time].tolist()
                        index = time_indices[0]
                        start_index = max(0, index - lookback)
                        lanes = data['Number of Lanes'].values[0]
                        data = df.loc[start_index:index]
                        flow = data.iloc[:,19:19 + lanes].sum(axis = 1)
                        #print('flow in loop', flow)
                        l += 1
        else:
            data = ild_data

    #print(data)
    return data

def trafficDataFinder_old(ild_name, date, time, lookback, rootpath):
    filename = os.path.join(rootpath, ild_name + '.csv')
    target_road = ilds[ilds['ILDs '] == ild_name]['Road id '].values[0]
    ild_list = ilds[ilds['Road id '] == target_road]
    
    # Check if the file exists
    if os.path.exists(filename):
        ild_data = pd.read_csv(filename, delimiter='\t')
        df = ild_data[ild_data['Date'] == date]
        time_indices = df.index[df['Time'] == time].tolist()
        index = time_indices[0]
        # Get data with lookback
        start_index = max(0, index - lookback)
        data = df.loc[start_index:index]
    
    # add files sorted the first one as the true one !!!
    else:
        # List all csv files in the rootpath
        files = glob.glob(os.path.join(rootpath, "*.csv"))
        file_list = []
        for file in files:
            if not ild_list[ild_list['ILDs '] == file[42:-4]].empty:
                file_list.append(file)
        # Find the closest ILD file
        closest_files = []
        files_with_diff = []
        closest_diff = float('inf')
        target_num = int(ild_name[:-1])
        target_letter = ild_name[-1]

        for file in file_list:
            if file[-5].isnumeric():
                file_num = int(file[42:-6])
                file_letter = file[-6]
            else:
                file_num = int(file[42:-5])
                file_letter = file[-5]
            
            #if ilds[ilds['ILDs '] == ild_name]['Road id ']
            if target_road == 'M40':
                target_letter = 'A'
            if file_letter == target_letter:
                diff = abs(target_num - file_num)
                files_with_diff.append((diff, file))
                # if diff < closest_diff:
                #     closest_diff = diff
                #     closest_files.append((diff, file))
                    # closest_file = file
        files_with_diff.sort(key=lambda x: x[0])
        sorted_files = [file for _, file in files_with_diff]
        #closest_files = sorted(closest_files, key=lambda x: x[0])[:30]

        if closest_files is None:
            raise FileNotFoundError(f"No file found matching ILD name {ild_name}")
        
        ild_data = pd.read_csv(sorted_files[0], delimiter='\t')
        k=0
        while ild_data.empty:
            k += 1
            if k == len(sorted_files):
                ild_data = pd.DataFrame(np.nan, index=[0, 1, 2, 3, 4], columns=ild_data.columns)
                print('breaking')
                break
            ild_data = pd.read_csv(sorted_files[k], delimiter='\t')
        print('ild_data', ild_data)
        if not ild_data.empty:
            # Filter data for the specific date
            df = ild_data[ild_data['Date'] == date]

            # Find the specific time index
            time_indices = df.index[df['Time'] == time].tolist()
            # print('time_indices', time_indices)
            if not time_indices:
                # data = pd.DataFrame(np.nan, index=[0, 1, 2, 3, 4], columns=ild_data.columns)
                raise ValueError(f"No data found for time {time} on date {date}")
                # find closest existing time ! 
                #time_indices 
            else:
                index = time_indices[0]
                
                # Get data with lookback
                start_index = max(0, index - lookback)
                data = df.loc[start_index:index]

                # check the data is not missing with positive flow
                lanes = data['Number of Lanes'].values[0]
                flow = data.iloc[:, 19:19 + lanes].sum(axis = 1)
                l = 1
                if np.sum(flow < 0) > 0:
                    while np.sum(flow < 0) > 1:
                        ild_data = pd.read_csv(sorted_files[l], delimiter='\t')
                        df = ild_data[ild_data['Date'] == date]
                        time_indices = df.index[df['Time'] == time].tolist()
                        index = time_indices[0]
                        start_index = max(0, index - lookback)
                        lanes = data['Number of Lanes'].values[0]
                        data = df.loc[start_index:index]
                        flow = data.iloc[:,19:19 + lanes].sum(axis = 1)
                        #print('flow in loop', flow)
                        l += 1
        else:
            data = ild_data

    #print(data)
    return data


#datatest = trafficDataFinder('2368J', '21/05/21', '12:46', 5, '../My_PhD-main/50May21_data/50May21_LDs/')

def trafficDataFinder_oldold(ild_name, date, time, lookback, rootpath):
    filename = rootpath + ild_name + '.csv'
    # check filename exists
    if exists(filename):
        ild_data = pd.read_csv(filename, delimiter='\t')
    else:
        # extract all filenames, then find the closest number with same letter
        files = glob.glob(rootpath + "*.csv")
        ild = []
        for file in files:
            if file[-5] == ild_name[-1]:
                ild.append(int(ild_name[:-1]) - int(file[-9:-5]))
        ind = np.argmin(ild)
        ild_data = pd.read_csv(files[ind], delimiter='\t')
    print(ild_data)
    df = ild_data[ild_data['Date'] == date]
    index = df[df['Time'] == time].index[0]
    data = df.loc[index-lookback:index, :]
    print(data)
    # data = df[df['Time'] == time]
    return data

# data = trafficDataFinder('3705A', '21/05/21', '09:00', 5, '../My_PhD-main/50May21_data/50May21_LDs/')
# print(data)

# how to store/load

