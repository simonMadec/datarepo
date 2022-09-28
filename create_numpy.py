

from pathlib import Path
import numpy as np
import rasterio
import os
from pathlib import Path
import random
from glob import glob
from util_patch_new_numba import ExtractPatch
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst

import faulthandler
faulthandler.enable()

root_data = "/home/simon/DATA/land_use_classification/data"

# ref = "150cm" 
# ref = "10m"
for dd in ["data_dordogne_origin","data_reunion_origin"]:
    for ref in ["150cm","10m"]:
    
        # _out150cmGSD bc it is the reference
        if dd == "data_ToyDordogne":
            ee = "data_dordogne_origin"
        elif dd=="data_ToyReunion":
            ee = "data_reunion_origin"
        else: 
            ee = dd

        #   ~#~~###   to delete
        if ref =="10m":
            gt_id_file = Path(root_data) /  f"{ee}" / "Ground_truth" / "GT_SAMPLES_Id.tif"
            gt_label_file = Path(root_data) /  f"{ee}" / "Ground_truth" / "GT_SAMPLES_Code.tif"
        else:
            gt_id_file = Path(root_data) /  f"{ee}" / "Ground_truth" / "GT_SAMPLES_Id_150cmGSD.tif"
            gt_label_file = Path(root_data) /  f"{ee}" / "Ground_truth" / "GT_SAMPLES_Code_150cmGSD.tif"

        S1_list_path = [glob(f"{root_data}/{dd}/Sentinel-1/{x}_ASC_CONCAT_S1.tif")[0] for x in ["VV","VH"]]
        S2_list_path =  [glob(f"{root_data}/{dd}/Sentinel-2/{x}_*GAPF.tif")[0] for x in ["B2","B3","B4","B8","NDVI","NDWI"]]
        SpotMs_list_path = [glob(f"{root_data}/{dd}/Spot-MS/*.tif")[0]]
        SpotP_list_path = [glob(f"{root_data}/{dd}/Spot-P/*.tif")[0]]

        path_out = str(Path(root_data) / f"{dd}_{ref}")
        Path(path_out).mkdir(parents=True, exist_ok=True)
        # temp = {}
        # for x in S1_list_path + S2_list_path + SpotMs_list_path + SpotP_list_path:
        #     with rasterio.open (x) as ds :
        #         array = ds.read()
        #         if "Sentinel-1" in x:
        #             array = -10*np.log(array)
        #         temp[str(Path(x).stem) ] = {}
        #         temp[str(Path(x).stem) ]["min"] = np.min(array)
        #         temp[str(Path(x).stem) ]["max"] = np.max(array)
        #         print(str(Path(x).stem) , np.min(array), np.max(array))
        # temp[dd] = dd

        # string= str(Path(path_out) / f"{dd}_{ref}_norm_info.npy")
        # print(f"saving .npy to : {string}")
        # np.save(string, temp)
        
        ExtractPatch(path_out,gt_id_file,gt_label_file,S1_list_path,S2_list_path,SpotMs_list_path[0],SpotP_list_path[0])    