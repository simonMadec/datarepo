import rasterio
import numpy as np
from rasterio.windows import from_bounds
from  scipy import ndimage
from pathlib import Path
import tqdm
import rasterio
from rasterio.transform import Affine as A

def read_S2 (lst,bounds):
    array = []
    for ts in lst :
        with rasterio.open (ts) as ds :
            print(ts)
            if not bounds:
                ts_array = ds.read()
            else:
                ts_array = ds.read(window=from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], transform=ds.profile['transform']))
            ts_array = np.moveaxis(ts_array,0,-1)
            ts_array = (ts_array - np.min(ts_array))/(np.max(ts_array) - np.min(ts_array))
            # cellsize = ds.profile['transform'][0]

            if ts_array.shape[2] > np.min([ts_array.shape[1],ts_array.shape[0]]):
                print("possible mismatch between height width ..")
            
            #update meta information with bounds
            new_transform = ds.profile['transform']
            if not bounds:
                print(f"we take all images for {ts}")
            else:
                new_transform= A(new_transform[0],new_transform[1],bounds[0],new_transform[3],new_transform[4],bounds[3])
            meta_data_dict = {"width": ts_array.shape[1], "height": ts_array.shape[0], "transform": new_transform} # tod ochqnge here the qffine

            kwds = ds.profile
            kwds.update(**meta_data_dict) #change affine

        array.append(ts_array)
        ts_array=None
    try:
        array = np.stack(array,axis=-1)
    except:
        array = np.dstack(array)
    array = array.reshape(array.shape[0],array.shape[1],np.prod(array.shape[2:]))

    with rasterio.Env(): 
        profile = kwds
        profile["count"]= array.shape[2]
        with rasterio.open('example4254542544.tif','w',**profile) as ds_out:
            ds_out.write(np.moveaxis(array,2,0))
    return array,ds_out
    # ds_out = rasterio.open(lst[0],'w' ,**kwds)
        

def read_S1 (lst,bounds):
    array_log = []
    for ts in lst :
        with rasterio.open (ts) as ds :
            if not bounds:
                ts_array = ds.read()
            else:
                ts_array = ds.read(window=from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], transform=ds.profile['transform']))
            # ts_array = ds.read()
            ts_array = np.moveaxis(ts_array,0,-1)
            ts_array_log = -10*np.log(ts_array)
            ts_array_log = (ts_array_log - np.min(ts_array_log))/(np.max(ts_array_log) - np.min(ts_array_log))

            if ts_array.shape[2] > np.min([ts_array.shape[1],ts_array.shape[0]]):
                print("possible mismatch between height width ..")

            #update meta information with bounds
            new_transform = ds.profile['transform']
            if not bounds:
                print(f"we take all images for {ts}")
            else:
                new_transform= A(new_transform[0],new_transform[1],bounds[0],new_transform[3],new_transform[4],bounds[3])
            meta_data_dict = {"width": ts_array.shape[1], "height": ts_array.shape[0], "transform": new_transform} # tod ochqnge here the qffine

            kwds = ds.profile
            kwds.update(**meta_data_dict) #change affine

        array_log.append(ts_array_log)
        ts_array_log=None
    array = np.dstack(array_log)
    array = array.reshape(array.shape[0],array.shape[1],np.prod(array.shape[2:]))

    with rasterio.Env(): 
        profile = kwds
        profile["count"]= array.shape[2]
        with rasterio.open('example4542423543.tif','w',**profile) as ds_out:
            ds_out.write(np.moveaxis(array,2,0))

    return array,ds_out

def read_ms_pan (ts, bounds):
    # array = []
    with rasterio.open (ts) as ds :
        if not bounds:
            ts_array = ds.read()
        else:
            ts_array = ds.read(window=from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], transform=ds.profile['transform']))
        ts_array = np.moveaxis(ts_array,0,-1)
        ts_array = (ts_array - np.min(ts_array))/(np.max(ts_array) - np.min(ts_array))
        cellsize = ds.profile['transform'][0]

        if ts_array.shape[2] > np.min([ts_array.shape[1],ts_array.shape[0]]):
            print("possible mismatch between height width ..")
        #update meta information with bounds
        new_transform = ds.profile['transform']

        if not bounds:
            print(f"we take all images for {ts}")
        else:
            new_transform= A(new_transform[0],new_transform[1],bounds[0],new_transform[3],new_transform[4],bounds[3])

        meta_data_dict = {"width": ts_array.shape[1], "height": ts_array.shape[0], "transform": new_transform} # tod ochqnge here the qffine
        kwds = ds.profile
        kwds.update(**meta_data_dict) #change affine

    array = ts_array.reshape(ts_array.shape[0],ts_array.shape[1],np.prod(ts_array.shape[2:]))
    with rasterio.Env(): 
        profile = kwds
        profile["count"]= array.shape[2]
        with rasterio.open('example24545.tif','w',**profile) as ds_out:
            ds_out.write(np.moveaxis(array,2,0))
    return array, ds_out

def latlo2pix(raster_src, lat, lon):
    py, px = raster_src.index(lon, lat)
    return py, px

def select_interest(A,path_out):
    if Path(path_out + "_gt_id_crp_select.npy").is_file():
        A_new= np.load(path_out + "_gt_id_crp_select.npy")
    else:
        Ae = ndimage.binary_erosion(A).astype(A.dtype)
        A_new = np.zeros(A.shape)
        A_filtered = A[:]
        A_filtered[Ae==0]=0
        unique = np.unique(A_filtered)
        for x in tqdm.tqdm(unique[unique!=0], desc = "select the id for down sampling"):    
            # compute the area of the considerd objects
            tot =  np.sum(A==x) #ce can avoid this by doing count..
            if tot < 66: # we keep small object
                temp = np.zeros(A.shape)
                temp[A==x] = A[A==x] 
                A_new = A_new + temp
            else:
                #not a small object we are going to select some parts of it
                #we grab the indexes of the ones
                #take filterd becasue we want to increase the window by +1
                xi,yi = np.where(A_filtered == x)
                #we chose one index randomly
                i = np.random.choice(len(xi),size=int(0.15*len(xi)),replace=False)
                # we increment the new matrix with +1 +1
                temp = np.zeros(A.shape)
                temp[xi[i] & xi[i]+1 ,yi[i] & yi[i]+1] = A_filtered[xi[i] & xi[i]+1 ,yi[i] & yi[i]+1] 
                A_new = A_new + temp
        np.save(path_out + "_gt_id_crp_select.npy", A_new)
    return A_new


def get_image_latlong(raster_path: str):
    with rasterio.open(raster_path) as src:
        band1 = src.read(1)
        print('Ref image has shape', band1.shape)
        height = band1.shape[0]
        width = band1.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        print('get lat and long info ..')
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons= np.array(xs)
        lats = np.array(ys)
        return lats, lons