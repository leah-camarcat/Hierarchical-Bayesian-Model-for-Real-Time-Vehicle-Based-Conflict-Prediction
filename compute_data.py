import numpy as np
import pandas as pd
import math
from utils import ILD_finder, trafficDataFinder, find_motorway, find_motorway_and_direction, calculate_bearing
import haversine
import geopandas as gpd
import warnings 

ilds = pd.read_csv('ild_loc_ab.csv', decimal=',')
ilds = ilds.iloc[:, 2:]
#print('ilds 2', ilds)
data_veh = pd.read_csv('vehicle_tracking_results_all_2024_egoveh3.csv')
data_veh = data_veh.iloc[:,]
#print(data_veh)

# split in scenes: 
scene = []
conflictTimesVideo = [[13*60+12, 23*60+34, 32*60+48, 35*60+15, 41*60+20, 48*60+47],
                      [30*60+16, 30*60+47, 31*60+28, 37*60+37, 46*60+42],
                      [18*60+35, 19*60+26, 20*60+46, 27*60+29, 28*60+28, 28*60+58, 30*60+45, 34*60+50, 35*60+2],
                      [61, 240+56, 352, 364, 563, 651, 12*60+32, 38*60+15, 39*60+2, 39*60+46, 43*60+35, 44*60+19, 45*60+56, 46*60+55, 48*60+30, 49*60+14, 49*60+52, 54*60+59],
                      [25, 89, 110, 154, 167, 21*60+29, 11*60+19, 22*60+9, 30*60+27, 33*60+14],
                      [405, 7*60+10, 644, 13*60+3, 19*60+42, 30*60+1, 30*60+8, 31*60+16, 33*60+21, 34*60+32, 40*60+10, 46*60+58, 51*60+13, 52*60+53],
                      [17*60+47, 21*60+11, 21*60+47, 24*60+11, 25*60+12, 29*60+25, 30*60+51, 31*60+29, 33*60+23, 50*60+40]]
conflict_number_before = 0
ordered_data = pd.DataFrame(columns=data_veh.columns)
for vid in data_veh.video.unique():
    vid_data = data_veh[data_veh.video == vid]
    for i in range(len(vid_data)):
        video_frame = vid_data.Frame.iloc[i]
        conflicttimes = np.array(conflictTimesVideo[vid])
        # conflicttimes = conflicttimes[np.nonzero(conflicttimes)]
        conflicttimessec = (conflicttimes-10)*5
        diff = int(video_frame) - conflicttimessec
        pos_diff = diff[diff>-0.0000001]
        scene.append(int(np.where(diff == int(pos_diff[-1]))[0][0]) + conflict_number_before)
    conflict_number_before = conflict_number_before + len(conflicttimes)
    ordered_data = pd.concat((ordered_data, vid_data), ignore_index=True)

ordered_data['Scene'] = pd.Series(scene)
data_veh = ordered_data
#print('number of scenes found: ', len(ordered_data['Scene'].unique()))

#for vid in ordered_data.video.unique():
    #vid_data = ordered_data[ordered_data.video == vid]
    #print('scene in video', vid, ':', np.unique(vid_data.Scene))
    #if len(np.unique(vid_data.Scene))==len(conflictTimesVideo[vid]):
        #print('True')

lookback = 4
rootpath = '../My_PhD-main/50May21_data/3050May21_LDs/'
date = '21/05/21'
roads = gpd.read_file("HAPMS_transformed/hapms_transformed.shp")
# roads = roads.set_crs(epsg=27700)  # Set the original CRS if undefined

motorways = roads[roads['roa_number'].str.startswith('M')]
traff_data = pd.DataFrame(columns=['ILD', 'Number of Lanes', 'Flow', 'Speed', 'Speed std','Occupancy', 'Headway', 'target_ID'])

distances = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for vid in data_veh.video.unique():
        vid_data = data_veh[data_veh.video==vid].copy()
        print('number of scenes in the video:', len(vid_data.Scene.unique()))
        for scene in vid_data.Scene.unique():
            scene_data = vid_data[vid_data.Scene==scene].copy()
            # find two different gps readings in order (with frames)
            for i in range(len(scene_data)-1):
                if scene_data.egolat.iloc[i+1] != scene_data.egolat.iloc[i]: #and (scene_data.Frame.iloc[i+1] - scene_data.Frame.iloc[i]>0):
                    lat1 = scene_data.iloc[i, 8]
                    lon1 = scene_data.iloc[i, 9]
                    lat2 = scene_data.iloc[i+1, 8]
                    lon2 = scene_data.iloc[i+1, 9]
                    if scene_data.Frame.iloc[i+1]>scene_data.Frame.iloc[i]:
                        motorway, direction = find_motorway_and_direction(lat1, lon1, lat2, lon2, motorways)
                    else:
                        motorway, direction = find_motorway_and_direction(lat2, lon2, lat1, lon1, motorways)
                    if motorway=='M1' and direction == 'Southbound':
                        dir = 'B'
                        ild_order = 'decrease'
                    if motorway=='M40' and direction == 'Westbound':
                        dir = 'A'
                        ild_order = 'increase'
                    if motorway=='M25' and direction == 'Westbound':
                        dir = 'B'
                        ild_order = 'decrease'
            ild_target = ILD_finder(ilds, lat1, lon1, dir)
            # print('ild', ild_target)
            time = data_veh.iloc[i, 10]
            hours = int(math.modf(time)[1])
            minutes = math.trunc(math.modf(time)[0] * 60)
            timestr = f"{hours}:{str(minutes).zfill(2)}"
            traff, dist = trafficDataFinder(ild_target[1], date, timestr, lookback, rootpath, dir, ild_order, motorway, lat1, lon1)
            distances.append(dist)
            # scene split
            # print(traff)
            lanes = traff['Number of Lanes']  # keep only this number of columns for each variable
            L = lanes.values[0]
            # remove from traff the lanes that aren't relevant, than average everything except sum for flow
            ild_selected = traff.iloc[:, 0]
            flow = traff.iloc[:,19:19+L].sum(axis = 1)
            speed = traff.iloc[:,12:12+L].mean(axis = 1)
            speedstd = traff.iloc[:,12:12+L].std(axis = 1)
            occ = traff.iloc[:, 26:26+L].mean(axis = 1)
            headway = traff.iloc[:, 33:33+L].mean(axis = 1)
            roadid = pd.DataFrame([ild_target[0] + '/' + ild_target[1]]*5)
            roadid.index = traff.index
            traffd = pd.concat((ild_selected, lanes, flow, speed, speedstd, occ, headway, roadid), axis = 1)
            traffd.columns = ['ILD', 'Number of Lanes', 'Flow', 'Speed', 'Speed std','Occupancy', 'Headway', 'target_ID']
            print('vid', vid, 'scene', scene)
            traff_data = pd.concat((traff_data, traffd), axis = 0)

print(len(traff_data))
traff_data.to_csv('traffic_vehicle_december2024_2.csv')
print(traff_data)