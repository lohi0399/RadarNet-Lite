# Copyright (c) Fangqiang Ding. All Rights Reserved

import os
import json
import ujson
import numpy as np
from concurrent import futures
import copy
from time import *
from joblib import Parallel, delayed
import multiprocessing
import sys
import concurrent.futures
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from pointcloud import LidarPointCloud2 as LidarPointCloud
from pointcloud import RadarPointCloud2 as RadarPointCloud
# DATA_PATH = '/home/toytiny/Desktop/RadarNet/data/nuscenes/'
DATA_PATH = r'C:\Users\vgandham\OneDrive - Delft University of Technology\Desktop\Perception\radaar\datasets\v1.0-mini'
OUT_PATH_PC = DATA_PATH + 'voxel_representations/'
OUT_PATH_AN = DATA_PATH + 'annotations/'
SPLITS = {
          #'mini_val': 'v1.0-mini',
          'mini_train': 'v1.0-mini',
          }
DEBUG = False
CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}

SENSOR_ID = {'LIDAR_TOP': 1,'RADAR_FRONT': 2, 'RADAR_FRONT_LEFT': 3, 
  'RADAR_FRONT_RIGHT': 4, 'RADAR_BACK_LEFT': 5, 
  'RADAR_BACK_RIGHT': 6}

RADAR_LIST = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 
  'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 
  'RADAR_BACK_RIGHT']

LIDAR_LIST=['LIDAR_TOP']

#Put lidar in front of radar
USED_SENSOR=['LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 
  'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 
  'RADAR_BACK_RIGHT']

NUM_SWEEPS_LIDAR = 1
NUM_SWEEPS_RADAR = 6

#suffix1 = '_{}sweeps'.format(NUM_SWEEPS) if NUM_SWEEPS > 1 else ''
#OUT_PATH = OUT_PATH + suffix1 + '/'

ATTRIBUTE_TO_ID = {
  '': 0, 'cycle.with_rider' : 1, 'cycle.without_rider' : 2,
  'pedestrian.moving': 3, 'pedestrian.standing': 4, 
  'pedestrian.sitting_lying_down': 5,
  'vehicle.moving': 6, 'vehicle.parked': 7, 
  'vehicle.stopped': 8}
side_range=(-50, 50) 
fwd_range=(-50, 50)
height_range = (-3,5)
res_height=0.25
res_wl = 0.15625
num_features=int((height_range[1]-height_range[0])/res_height);
num_x=int((fwd_range[1]-fwd_range[0])/res_wl);
num_y=int((side_range[1]-side_range[0])/res_wl);


                        
def voxel_generate_lidar(points,side_range,fwd_range, height_range, res_wl, res_height):
    
    num_features=int((height_range[1]-height_range[0])/res_height);
    num_x=int((fwd_range[1]-fwd_range[0])/res_wl);
    num_y=int((side_range[1]-side_range[0])/res_wl);

    feature_list=[]
    
    x_points = points[0,:]
    y_points = points[1,:]
    z_points = points[2,:]
    
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    h_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filt = np.logical_and(np.logical_and(f_filt, s_filt),h_filt)
    indices = np.argwhere(filt).flatten()
    points=points[:,indices]
    
    x_points = points[0,:]
    y_points = points[1,:]
    z_points = points[2,:]
    
    # SHIFT to the BEV view
    x_img = x_points + fwd_range[1] 
    y_img = -(y_points + side_range[0])
    z_img = z_points-height_range[0]
    
    # index of points in features
    z_ind=np.floor((z_img)/res_height)
    x_ind=np.floor((x_img)/res_wl)
    y_ind=np.floor((y_img)/res_wl)
    
    # center of voxels in features
    z_c=(z_ind+1/2)*res_wl
    x_c=(x_ind+1/2)*res_wl
    y_c=(y_ind+1/2)*res_wl
    
    features=np.zeros((num_features,num_y,num_x))
    
    for ind in range(np.size(points,1)):
        
        features[int(z_ind[ind]),int(y_ind[ind]),int(x_ind[ind])]+=(1-abs(x_img[ind]-x_c[ind]/(res_wl/2)))* \
            (1-abs(y_img[ind]-y_c[ind]/(res_wl/2)))*(1-abs(z_img[ind]-z_c[ind]/(res_height/2)))
        
        
    return features.tolist()

def voxel_generate_radar(points,side_range,fwd_range, res_wl):
    
    num_x=int((fwd_range[1]-fwd_range[0])/res_wl);
    num_y=int((side_range[1]-side_range[0])/res_wl);
    
    feature= np.zeros([num_y, num_x],dtype=np.float16)
        
    
    x_points = points[0,:]
    y_points = points[1,:]
    dy_prop  = points[3,:]
            
            
    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis car coordinates
        
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filt = np.logical_and(f_filt, s_filt)
        
    if not any(filt):
        return feature
        
    indices = np.argwhere(filt).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    dy_prop  = dy_prop[indices]
        

    # SHIFT to the BEV view
    x_img = x_points + fwd_range[1] 
    y_img = -(y_points + side_range[0])
    
     
        
    # calculate the value for each voxel in the feature
    for j in range(0,num_x):
        x_filt=np.logical_and((x_img > j*res_wl), (x_img < (j+1)*res_wl))
        if not any(x_filt):
            feature[:,j]=0
            continue
        else:
            # abandon those points not in this column to save time 
            indices=np.argwhere(x_filt).flatten()
            x_img_cl=x_img[indices]
            y_img_cl=y_img[indices]
            dy_prop_cl=dy_prop[indices]
            for k in range(0,num_y):
                y_filt=np.logical_and((y_img_cl > k*res_wl), (y_img_cl < (k+1)*res_wl))
                if not any(y_filt):
                    feature[k,j]=0
                else:
                    indices=np.argwhere(y_filt).flatten()
                    dy_prop_v=dy_prop_cl[indices]
                    if any(dy_prop_v==0) or any(dy_prop_v==2) or any(dy_prop_v==6):
                        feature[k,j]=1
                    else:
                        feature[k,j]=-1
                    
                        
        
    feature=feature.tolist()
        
        
    return feature

def get_radar_target(points,times, side_range,fwd_range,height_range,res_wl):
    
    x_points = points[0,:]
    y_points = points[1,:]
    z_points = points[2,:]
    dy_prop=points[3,:]
    vel_x = points[8,:]
    vel_y = points[9,:]
    times=times[0,:]   
            
    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis car coordinates
        
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    h_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filt = np.logical_and(np.logical_and(f_filt, s_filt),h_filt)
        
    indices = np.argwhere(filt).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    dy_prop=dy_prop[indices]
    times=times[indices]
    vel_x = vel_x[indices]
    vel_y = vel_y[indices]
    
    
    # SHIFT to the BEV view

    x_img = np.floor((x_points + fwd_range[1])/res_wl)
    y_img = np.floor(-(y_points + side_range[0])/res_wl)
    
    # get the motion information
    for i in range(0,len(dy_prop)):
        if dy_prop[i] in [0,2,6]:
            dy_prop[i]=1
        else:
            dy_prop[i]=0
         
    # Transfer to the velocity in BEV view
    vel_x=vel_x/res_wl
    vel_y=-vel_y/res_wl
     
    # Calculate the radial velocity towards the car in BEV view
    theta=np.arctan((x_img-(fwd_range[1]-fwd_range[0])/2)/(y_img-1e-4-(side_range[1]-side_range[0])/2))
    vel_r=(-vel_y*np.cos(theta)-vel_x*np.sin(theta))
    
    targets=[]
    # point cloud information 
    for i in range(0,len(x_points)):
        if np.isnan(vel_r[i]):
            vel_r[i]=0
        target = {'location': [x_img[i],y_img[i]],
                  'vel_r': vel_r[i],
                  'motion':dy_prop[i],
                  'time': times[i], 
                  }
        targets.append(target)
        
    return targets   
        
   
def point_exist_in_box(box,pcs): 
     return True
   
    

def main():
  if not os.path.exists(OUT_PATH_PC):
    os.mkdir(OUT_PATH_PC)
  # convert one spilt at one time  
  for split in SPLITS:
    
    data_path = DATA_PATH
    nusc = NuScenes(
      version=SPLITS[split], dataroot=data_path, verbose=True)
    out_path_pc = OUT_PATH_PC + split 
    out_path_an = OUT_PATH_AN + split
    if not os.path.exists(out_path_pc):
        os.makedirs(out_path_pc)
    if not os.path.exists(out_path_an):
        os.makedirs(out_path_an)
    categories_info = [{'name': CATS[i], 'id': i + 1} for i in range(len(CATS))]
    ret = {'pcs': [], 'annotations': [], 'categories': categories_info, 
           'scenes': [], 'attributes': ATTRIBUTE_TO_ID}
    
    num_scenes = 0
    num_pcs = 0
    num_anns = 0
    
    # A "sample" in nuScenes refers to a timestamp with 5 RADAR and 1 LIDAR keyframe.
    for sample in nusc.sample:
      scene_name = nusc.get('scene', sample['scene_token'])['name']
      if not (split in ['test']) and \
         not (scene_name in SCENE_SPLITS[split]):
         continue
      if sample['prev'] == '':
        print('scene_name', scene_name)
        num_scenes+= 1
        ret['scenes'].append({'id': num_scenes, 'file_name': scene_name})
        track_ids = {}
        # skip the first keyframe since it has no prev sweeps  
        continue
      
      # Load lidar points from files and transform them to car coordinate  
      pc_token = sample['data'][LIDAR_LIST[0]]
      pc_data = nusc.get('sample_data', pc_token)
      num_pcs += 1
      out_path_current=out_path_pc+'/'+'voxel_scenes-{}_pcs-{}'.format(num_scenes,num_pcs)
      out_path_annos=out_path_an+'/'+'anno_scenes-{}_pcs-{}.json'.format(num_scenes,num_pcs)
      if os.path.exists(out_path_current) and os.path.exists(out_path_annos):
           continue
      # Complex coordinate transform from Lidar to car
      sd_record = nusc.get('sample_data', pc_token)
      cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
      pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
      trans_matr = transform_matrix(
                 cs_record['translation'], Quaternion(cs_record['rotation']),
                 inverse=False)
      # velocity transform from car to global
      vel_global_from_car = transform_matrix(np.array([0,0,0]),
            Quaternion(pose_record['rotation']), inverse=False)
      
      
      print('Aggregating lidar data for sample:', num_pcs)   
      
      lidar_pcs, _ = LidarPointCloud.from_file_multisweep(nusc, 
                               sample, LIDAR_LIST[0], LIDAR_LIST[0], NUM_SWEEPS_LIDAR)
      
        ## Transform lidar point clound from sensor coordinate to car
      voxel_lidar=[]
      voxel_radar=[]
      for i in range(0, len(lidar_pcs)):
            lidar_pcs[i][:3, :] = trans_matr.dot(np.vstack((lidar_pcs[i][:3, :], 
                                   np.ones(lidar_pcs[i].shape[1]))))[:3, :]
        #     # Extract voxel presentations for a timestamp
            print('Extracting voxel representation for lidar sweep:', i+1) 
            # begin_time=time()
            voxel_lidar.append(voxel_generate_lidar(lidar_pcs[i],side_range,fwd_range,height_range,res_wl,res_height))
            # end_time=time()
            # runtime=end_time-begin_time
            # print(runtime)
          
   
      radar_pcs=[list([]) for i in range(0,NUM_SWEEPS_RADAR)]
      radar_times=[list([]) for i in range(0,NUM_SWEEPS_RADAR)]
      print('Aggregating radar data for sample:', num_pcs)  
      # Read radar points from files and tranpose them to car coordinate        
      for sensor_name in RADAR_LIST:
          
          current_radar_pcs, current_radar_times = RadarPointCloud.from_file_multisweep(nusc, 
                        sample, sensor_name, LIDAR_LIST[0], NUM_SWEEPS_RADAR)
          #print(len(current_radar_pcs))
          # Transpose radar point clound from lidar coordinate to car (in the above function, the velocity is automatically
          # transformed to the car coordinate)
          for i in range(0, len(current_radar_pcs)):
              current_radar_pcs[i][:3, :] = trans_matr.dot(np.vstack((current_radar_pcs[i][:3, :], 
                                    np.ones(current_radar_pcs[i].shape[1]))))[:3, :]
              # stack points from all five radar sensors
              if not len(radar_pcs[i]):
                  radar_pcs[i] = current_radar_pcs[i]
                  radar_times[i] = current_radar_times[i]
              else:
                  radar_pcs[i] = np.hstack((radar_pcs[i], current_radar_pcs[i]))    
                  radar_times[i] = np.hstack((radar_times[i], current_radar_times[i]))    
      radar_target=[]            
      for i in range(0,NUM_SWEEPS_RADAR): 
          #print('Extracting voxel representation for radar sweep:', i+1)  
          voxel_radar.append(voxel_generate_radar(radar_pcs[i],side_range,fwd_range,res_wl))
          #print('Getting radar target information from radar sweep', i+1)  
          radar_target=np.hstack((radar_target, get_radar_target(radar_pcs[i], radar_times[i], 
                                                                  side_range,fwd_range,height_range,res_wl)))
      #point cloud information 
      h_layers=int((height_range[1]-height_range[0])/res_height)
      num_chan=int(NUM_SWEEPS_RADAR+NUM_SWEEPS_LIDAR*h_layers)
      out_path_dir=out_path_pc+'/'+'voxel_scenes-{}_pcs-{}'.format(num_scenes,num_pcs)+'/'
      if not os.path.exists(out_path_dir):
        os.makedirs(out_path_dir)
      chan=0
      
      for ch in range(0,NUM_SWEEPS_RADAR):
         
          curr_img=np.array(voxel_radar[ch])
          curr_img=np.array(curr_img*127+128,dtype='uint8')# map [-1 1] to [1 255]
          curr_img_path=out_path_dir+'{}.jpg'.format(ch)
          cv2.imwrite(curr_img_path,curr_img)
          chan+=1
          # save the lidar data then
      for lid in range(0,NUM_SWEEPS_LIDAR):
          
          for lay in range(0,h_layers):
              curr_img=np.array(voxel_lidar[lid][lay])
              curr_img=np.array(curr_img*30,dtype='uint8')
              curr_img_path=out_path_dir+'{}.jpg'.format(chan)
              cv2.imwrite(curr_img_path,curr_img)
              chan+=1
      # pc_input = {'id': num_pcs,
      #              'scene_id': num_scenes,
      #              'scene_name': scene_name,
      #              'radar_feat': voxel_radar, 
      #              'lidar_feat': voxel_lidar,
      #              'radar_target': radar_target.tolist(),
      #              'timestap': sample['timestamp']/1e6,
      #              }
      radar_voxel_channel=len(voxel_radar)
      lidar_voxel_channel=len(voxel_lidar)
      print('Save {} voxel for {} sample in {} scene'.format(
          split, num_pcs, num_scenes))
      print('Lidar voxel channel: {}, Radar voxel channel: {}'.format(lidar_voxel_channel,radar_voxel_channel))
       #print('out_path', out_path_current)
      

      
      # ujson.dump(pc_input, open(out_path_current, 'w'))
      
      
      
      _,boxes,_ = nusc.get_sample_data(pc_token, box_vis_level=BoxVisibility.ANY)
     
      anns = []
      boxes_in=[]
      print('Aggregating annotations for sample:', num_pcs) 
      # Abandon boxes not in the detection region
      for box in boxes:
          #nusc.render_annotation(my_instance['first_annotation_token'])
          # Transform the boxes from sensor coordinate to car
          box.rotate(Quaternion(cs_record['rotation']))
          box.translate(cs_record['translation'])
          
          f_filt = np.logical_and(((box.center[0]-box.wlh[0]/2)>fwd_range[0]),((box.center[0]+box.wlh[0]/2)<fwd_range[1]))
          s_filt = np.logical_and(((box.center[1]-box.wlh[1]/2)>-side_range[1]),((box.center[1]+box.wlh[1]/2)<-side_range[0]))
          h_filt = np.logical_and(((box.center[2]-box.wlh[2]/2)>height_range[0]),((box.center[2]+box.wlh[2]/2)<height_range[1]))
          filt = np.logical_and(np.logical_and(f_filt, s_filt),h_filt)
          
    
          if filt:
              boxes_in.append(box)
      
      boxes_all=[]
      for box in boxes_in:
         exist_point=point_exist_in_box(box,lidar_pcs)
         if exist_point:
             boxes_all.append(box)
             
          
              
      for box in boxes_all:
          # Map the catergory to detection name
          det_name = category_to_detection_name(box.name)
          if det_name is None:
              continue
          num_anns += 1
          category_id = CAT_IDS[det_name]
          v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
          yaw = np.arctan2(v[1], v[0])
          vel = nusc.box_velocity(box.token)
          # get velocity in car coordinates
          vel = np.dot(np.linalg.inv(vel_global_from_car), 
              np.array([vel[0], vel[1], vel[2], 0], np.float32))
          vel=copy.deepcopy(vel[:2]) # only keep v_x, v_y
          center=copy.deepcopy(box.center[:2]) # only keep the p_x, p_y
          wl=copy.deepcopy(box.wlh[:2]) # only keep the width and length
          # project the object center, velocity and wl from car coordinate to BEV
          # SHIFT to BEV coordinate
          center[0] = np.floor((center[0] + fwd_range[1])/res_wl)
          center[1] = np.floor(-(center[1] + side_range[0])/res_wl)
          
          wl[0]=wl[0]/res_wl
          wl[1]=wl[1]/res_wl
          
          vel[0]=vel[0]/res_wl
          vel[1]=-vel[1]/res_wl
          
          if np.isnan(vel[0]) or np.isnan(vel[1]):
              vel[0]=0
              vel[1]=0
              
          sample_ann = nusc.get(
              'sample_annotation', box.token)
          instance_token = sample_ann['instance_token']
          if not (instance_token in track_ids):
              track_ids[instance_token] = len(track_ids) + 1
          attribute_tokens = sample_ann['attribute_tokens']
          attributes = [nusc.get('attribute', att_token)['name'] \
                        for att_token in attribute_tokens]
          att = '' if len(attributes) == 0 else attributes[0]
          if len(attributes) > 1:
              print(attributes)
              import pdb; pdb.set_trace()
          track_id = track_ids[instance_token]
          # annotations information 
          ann = {
              'id': num_anns,   # id for annotation instance 
              'pc_id': num_pcs,     # id for point cloud sample instance, the same as 'id': num_pcs in pcs
              'category_id': category_id,   # id for category of the object
              'dim': [wl[0],wl[1]],   # dimension in width, length 
              'location': [center[0], center[1]],   # object center location  
              'rotation_z': yaw,    # object pitch angle
              'track_id': track_id,     # id for track object
              'attributes': ATTRIBUTE_TO_ID[att],  # object attributes
              'velocity':[vel[0],vel[1]],  # object velocity in the car coordinate, BEV images
            }
          anns.append(ann)
          
      print('Save {} annos for {} sample in {} scene'.format(
              split, num_pcs, num_scenes))
      
      #print('out_path', out_path_annos)
      print('======')
      ujson.dump(anns, open(out_path_annos, 'w'))
          

SCENE_SPLITS = {
'mini_train':
    ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100'],
'mini_val':
    ['scene-0103', 'scene-0916'],    
}
    
if __name__ == '__main__':
   with concurrent.futures.ProcessPoolExecutor() as executor:
       executor.map(main())
  #main()
  #executor = futures.ThreadPoolExecutor(max_workers=3)
  #executor.map(main())
  #main()
    
    
    
    
    
    
    
    
    
    
    
    
