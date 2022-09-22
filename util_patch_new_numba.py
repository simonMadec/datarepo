import numpy as np
import rasterio
import os
from pathlib import Path
import random
from matplotlib import pyplot as plt
from  scipy import ndimage
from tqdm import tqdm 
from util_read import read_S1, read_S2, read_ms_pan, latlo2pix
import psutil
from time import time
from datetime import datetime
from numba import jit, njit
import pickle
from numba.typed import List
import timeit
from rasterio.windows import from_bounds
import faulthandler
from rasterio.transform import Affine as A
from numba_progress import ProgressBar
import json
import pandas as pd
import shutil 


@jit(nopython=True)
def toto(array,window,pixs1,patchs):
    # p0 = int(pixs1[0])
    # p1 = int(pixs1[1])
    ts1 = array[pixs1[0]-window:pixs1[0]+window+1,pixs1[1]-window:pixs1[1]+window+1]
    ts1 = np.expand_dims(ts1,0)
    patchs = np.concatenate((patchs,ts1), axis=0)
    return patchs

@jit(nopython=True)
def toto2(gt_id,xi,yi,gt_label,gt):
    ts0 = np.array((gt_id[xi,yi],gt_label[xi,yi],xi,yi)).reshape(-1,4)
    gt = np.concatenate((gt,ts0))
    return gt

@jit(nopython=True)
def loop_ind_numba(
    s1_patchs,
    s2_patchs,
    ms_patchs,
    pan_patchs,
    lpix,
    s1_array,
    s2_array,
    ms_array,
    pan_array,
    s1_window,
    ms_window,
    pan_window,
    gt,
    gt_id,
    gt_label):
    
    c_=-1

    for i_ in lpix:
        c_=c_+1
        pixs1 = lpix[c_][0]
        pixs2 = lpix[c_][1]
        pixms = lpix[c_][2]
        pixpan = lpix[c_][3]
        xi,yi = lpix[c_][4]
        
        if not (pixs1[0]-s1_window<0 or pixs1[1]-s1_window<0 or 1+pixs1[0]+s1_window+1>s1_array.shape[0]  or 1+pixs1[1]+s1_window+1>s1_array.shape[1] or \
            pixs2[0]-s1_window<0 or pixs2[1]-s1_window<0 or 1+pixs2[0]+s1_window+1>s2_array.shape[0]  or 1+pixs2[1]+s1_window+1>s2_array.shape[1] or \
            pixms[0]-ms_window<0 or pixms[1]-ms_window<0 or 1+pixms[0]+ms_window+1>ms_array.shape[0]  or 1+pixms[1]+ms_window+1>ms_array.shape[1] or \
            pixpan[0]-pan_window<0 or pixpan[1]-pan_window<0 or 1+pixpan[0]+pan_window+1>pan_array.shape[0]  or 1+pixpan[1]+pan_window+1>pan_array.shape[1]):
            
            gt = toto2(gt_id,xi,yi,gt_label,gt)
            s2_patchs = toto(s2_array,s1_window,pixs1,s2_patchs)
            s1_patchs = toto(s1_array,s1_window,pixs2,s1_patchs)
            ms_patchs = toto(ms_array,ms_window,pixms,ms_patchs)
            pan_patchs = toto(pan_array,pan_window,pixpan,pan_patchs)

    return gt,s1_patchs, s2_patchs, ms_patchs, pan_patchs


# @jit(nopython=True)
def ExtractPatch(path_out,gt_id_file,gt_label_file,lstS1,lstS2,ms,pan,s1_window=4,ms_window=4,pan_window=16,numfold=5):
    Path(path_out).mkdir(parents=True, exist_ok=True)
    print("read id file")

    # BoundingBox(left=497771.3400000001, bottom=6416616.860000001, right=506211.3400000001, top=6422996.860000001)
    # box for dordogne
    
    
    bb = [497771.3,  6416616.8, 506211.34, 6422996.86]
    bb = []

    ms_array, dms = read_ms_pan (ms,bb)
    ms_array = ms_array.reshape(ms_array.shape[0],ms_array.shape[1],np.prod(ms_array.shape[2:]))
    pan_array, dpan = read_ms_pan (pan,bb)
    pan_array = pan_array.reshape(pan_array.shape[0],pan_array.shape[1],np.prod(pan_array.shape[2:]))
    s1_array, ds1 = read_S1 (lstS1,bb)
    s1_array = s1_array.reshape(s1_array.shape[0],s1_array.shape[1],np.prod(s1_array.shape[2:])) #to do the reshape in the read function
    s2_array, ds2 = read_S2 (lstS2,bb)
    s2_array = s2_array.reshape(s2_array.shape[0],s2_array.shape[1],np.prod(s2_array.shape[2:]))
    # la 6455634.75 lo_486417.75

    print("reading gt_id_file")
    with rasterio.open(gt_id_file) as ds :
        if not bb:
            gt_id = ds.read(1)
        else:
            win = from_bounds(bb[0], bb[1], bb[2], bb[3], transform=ds.transform)
            gt_id = ds.read(1,window=win)

        print('Ref image has shape', gt_id.shape)
        height = gt_id.shape[0]
        width = gt_id.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        print("get lons lats ..")

        if not bb:
            xs, ys = rasterio.transform.xy(ds.transform, rows, cols)
        else:
            xs, ys = rasterio.transform.xy(A(ds.transform[0],ds.transform[1],bb[0],ds.transform[3],ds.transform[4],bb[3]), rows, cols)
        print("transform to coordinate")
        lons= np.array(xs,dtype=np.float32)
        lats = np.array(ys,dtype=np.float32)

    print("read label file")
    with rasterio.open(gt_label_file) as ds :
        if not bb:
            gt_label = ds.read(1)
        else:
            win = from_bounds(bb[0], bb[1], bb[2], bb[3], transform=ds.transform)
            gt_label = ds.read(1,window=win)      
    print("get unique of gt_label")
    classes = np.unique(gt_label)
    idx = {}
    for n_ in range(0,numfold):
        print(f"do split for fold {numfold}")
        train_id = []
        valid_id = []
        test_id = []

        for cl_ in classes:
            list_c = np.unique(gt_id[gt_label == cl_])
            np.random.shuffle(list_c)
            # split in 70% 15% 15%
            training, valid, test = list_c[:int(0.7*len(list_c))], list_c[int(0.7*len(list_c)):int(0.85*len(list_c))], list_c[int(0.85*len(list_c)):]
            train_id.extend(training.tolist())
            valid_id.extend(valid.tolist())
            test_id.extend(test.tolist())

        idx[n_] = (train_id,valid_id,test_id)

    print("finish to do train test valid")
    # with open('saved_dictionary_debug.pkl', 'wb') as f:
    #     pickle.dump(idx, f)
        
    # with open('saved_dictionary_debug.pkl', 'rb') as f:
    #     idx = pickle.load(f)

    splits = ['Training','Validation','Test']

    shutil.rmtree(path_out, ignore_errors=True)
    print("loop over the number of fold (5 by default)")
    for n_ in range(0,numfold): # loop over the number of fold 5 by default 
        for x_ in splits:

            Path( Path(path_out) / "Sentinel-1" / f"{x_}" / f"{n_}").mkdir(parents=True, exist_ok=True)
            Path( Path(path_out) / "Sentinel-2" / f"{x_}" / f"{n_}").mkdir(parents=True, exist_ok=True)
            Path( Path(path_out) / "Spot-MS" / f"{x_}" / f"{n_}").mkdir(parents=True, exist_ok=True)
            Path( Path(path_out) / "Spot-P" / f"{x_}" / f"{n_}").mkdir(parents=True, exist_ok=True)
            Path( Path(path_out) / "Ground_truth" / f"{x_}" / f"{n_}").mkdir(parents=True, exist_ok=True)

        for k_,_ in enumerate(idx[n_]): # loop over splits

            id = idx[n_][k_]
            path_out_list = f"{path_out}_{splits[k_]}"

            # if Path(f"{path_out_list}.npy").exists():
            #     lpix = list(np.load(f"{path_out_list}.npy"))
            # else:
            lpix= []
            for e_ in tqdm(id, desc= f"get id loc do for {splits[k_]} space use: {psutil.virtual_memory()[2]}"):
                ofint = gt_id==e_

                if np.sum(ofint)==0:
                    print("no object found")
                    continue

                if e_==0:
                    print("skipping id 0")
                    continue

                xi,yi = np.where(ndimage.binary_erosion(ofint))

                if len(xi)<66: # if large we select 15%
                    for i_ in range(0,len(xi)):
                        la_= lats[xi[i_],yi[i_]] 
                        lo_ = lons[xi[i_],yi[i_]]
                        lpix.append(np.array([latlo2pix(ds1,la_,lo_),latlo2pix(ds2,la_,lo_),latlo2pix(dms,la_,lo_),latlo2pix(dpan,la_,lo_),(xi[i_],yi[i_])],dtype=np.int32))
                else:
                    ind =  np.random.choice(len(xi),size=int(0.05*np.sqrt(len(xi)*len(yi))),replace=False)
                    for i_ in range(0,len(ind)):
                        la_= lats[xi[i_],yi[i_]] 
                        lo_ = lons[xi[i_],yi[i_]]
                        lpix.append(np.array([latlo2pix(ds1,la_,lo_),latlo2pix(ds2,la_,lo_),latlo2pix(dms,la_,lo_),latlo2pix(dpan,la_,lo_),(xi[i_],yi[i_])],dtype=np.int32))

                        la_= lats[xi[i_]+1,yi[i_]] 
                        lo_ = lons[xi[i_]+1,yi[i_]]
                        lpix.append(np.array([latlo2pix(ds1,la_,lo_),latlo2pix(ds2,la_,lo_),latlo2pix(dms,la_,lo_),latlo2pix(dpan,la_,lo_),(xi[i_]+1,yi[i_])],dtype=np.int32))

                        la_= lats[xi[i_]+1,yi[i_]+1] 
                        lo_ = lons[xi[i_]+1,yi[i_]+1]
                        lpix.append(np.array([latlo2pix(ds1,la_,lo_),latlo2pix(ds2,la_,lo_),latlo2pix(dms,la_,lo_),latlo2pix(dpan,la_,lo_),(xi[i_]+1,yi[i_]+1)],dtype=np.int32))

                        la_= lats[xi[i_],yi[i_]+1] 
                        lo_ = lons[xi[i_],yi[i_]+1]
                        lpix.append(np.array([latlo2pix(ds1,la_,lo_),latlo2pix(ds2,la_,lo_),latlo2pix(dms,la_,lo_),latlo2pix(dpan,la_,lo_),(xi[i_],yi[i_]+1)],dtype=np.int32))

                nparray = np.asarray(lpix)
                np.save(f"{path_out_list}",nparray)
                

                # with open(path_out_list, "w") as fp:
                #     json.dump(lpix, fp)   
                # on mixe lpix pour l'entrainement/test/valid ..
                
            num_iterations = len(lpix)
            print(f"we have a number of patches of {num_iterations}")
            random.shuffle(lpix)
            start_time = datetime.now()
            rr = range(0, len(lpix), 256)
            for i0,i in enumerate(tqdm(rr,desc=f"stacking batch of array for {path_out}")):
                gt = np.array([]).reshape(0,4)
                s1_patchs = np.array([]).reshape(0,s1_window*2+1,s1_window*2+1,s1_array.shape[2])
                s2_patchs= np.array([]).reshape(0,s1_window*2+1,s1_window*2+1,s2_array.shape[2])
                ms_patchs= np.array([]).reshape(0,ms_window*2+1,ms_window*2+1,ms_array.shape[2])
                pan_patchs= np.array([]).reshape(0,pan_window*2+1,pan_window*2+1,pan_array.shape[2])
                laliste256 = List(lpix[i:i+256])
                gt_out, s1_patchs_out, s2_patchs_out, ms_patchs_out, pan_patchs_out  = loop_ind_numba(s1_patchs,s2_patchs,ms_patchs,pan_patchs,laliste256,s1_array,s2_array,ms_array,pan_array,s1_window,ms_window,pan_window,gt,gt_id,gt_label)       

                gt_out[:,1]= gt_out[:,1]-1

                np.save(Path(path_out) / 'Ground_truth' / f"{splits[k_]}" / f"{n_}"/ f"Ground_truth_{splits[k_]}_split_{i0}.npy",gt_out)
                np.save(Path(path_out) / 'Sentinel-1' / f"{splits[k_]}" / f"{n_}" / f"Sentinel-1_{splits[k_]}_split_{i0}.npy",s1_patchs_out)
                np.save(Path(path_out) / 'Sentinel-2' / f"{splits[k_]}" / f"{n_}" / f"Sentinel-2_{splits[k_]}_split_{i0}.npy",s2_patchs_out)
                np.save(Path(path_out) / 'Spot-MS' / f"{splits[k_]}" / f"{n_}" / f"Spot-MS_{splits[k_]}_split_{i0}.npy",ms_patchs_out)
                np.save(Path(path_out) / 'Spot-P' / f"{splits[k_]}" / f"{n_}" / f"Spot-P_{splits[k_]}_split_{i0}.npy",pan_patchs_out)



