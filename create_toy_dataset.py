import os
import glob
from osgeo import gdal
from pathlib import Path
import rasterio
from rasterio.windows import from_bounds
import numpy as np

# upper_left_x = 339715.79
# upper_left_y = 7641853.86
# lower_right_x = 346683.90
# lower_right_y = 7637158.79

# window = (upper_left_x,upper_left_y,lower_right_x,lower_right_y)
# for x in glob.glob("data_reunion/Sentinel-1/*.tif"):
#     with rasterio.open(x) as dscheck :
#         ts_array = dscheck.read()
#         print("lorigine", np.unique(ts_array), ts_array.shape)
#     print(Path(x).stem)
#     print(os.path.getsize(x))
#     ds = gdal.Open(x)
#     # driver = gdal.GetDriverByName('GTiff') 
#     # outDs = driver.Create(str(Path("ToyReunion") / x), GDT_Float32)

#     out = gdal.Translate(str(Path("data_ToyReunion") / Path(x).parent.stem / Path(x).name),ds, projWin = window)
#     ds = None
#     out = None
#     with rasterio.open(str(Path("data_ToyReunion") / Path(x).parent.stem / Path(x).name)) as dscheck :
#         ts_array = dscheck.read()
#         print("on est la", ts_array.shape)

upper_left_x = 497771.34 # 297060.431
upper_left_y = 6422996.86 # 4972732.834
lower_right_x = 506213.76 # 306017.492
lower_right_y = 6416615.44 # 4964661.193
# upper_left_x = 0.441549
# upper_left_y = 44.878516
# lower_right_x = 0.536535
# lower_right_y = 44.822807

for x in glob.glob("data_dordogne/*/*.tif"):
    
    if (Path(x).parent.stem == "Sentinel-2") & ("_GAPF" not in x):
        print("not processing")
    # elif (Path(x).parent.stem == "Sentinel-1") :
    #     print("not processing")
    else:
        print(f"processing {x}")
        with rasterio.open(x) as src :
            # read only the window of interest from the raster
            aoi_window = from_bounds(upper_left_x, lower_right_y, lower_right_x, upper_left_y, src.transform)
            aoi_arr = src.read(window=aoi_window)

            # create new transform and update the profile to attach while saving the cropped raster
            aoi_transform = src.window_transform(aoi_window)
            aoi_prof = src.profile

            aoi_prof.update({
                'height':  aoi_arr.shape[1],
                'width': aoi_arr.shape[2],
                'transform': aoi_transform
            })

            # write the aoi to file
            op_path = str(Path("data_ToyDordogne") / Path(x).parent.stem / Path(x).name)
            with rasterio.open(op_path, 'w', **aoi_prof) as dst:
                dst.write(aoi_arr)
                
                
                        #     breakpoint()

        #     ts_array = dscheck.read()
        #     print("lorigine", np.unique(ts_array), ts_array.shape)
        # print(Path(x).stem)
        # print(os.path.getsize(x))
        # breakpoint()
        # ds = gdal.Open(x)
        # ###driver = gdal.GetDriverByName('GTiff') 
        # ###outDs = driver.Create(str(Path("ToyReunion") / x), GDT_Float32)
        # out = gdal.Translate(str(Path("data_ToyDordogne") / Path(x).parent.stem / Path(x).name),ds, projWin = window)
        # ds = None
        # out = None
        # with rasterio.open(str(Path("data_ToyDordogne") / Path(x).parent.stem / Path(x).name)) as dscheck :
        #     ts_array = dscheck.read()
        #     print("Lecture : ", ts_array.shape)
        #     print("Lecture unique: ",  np.unique(ts_array))
        #     breakpoint()
        # breakpoint()
            
for x in glob.glob("data_ToyDordogne/*/*tif"):
    with rasterio.open(x) as src:
        print(f"reading.. {Path(x).stem}")
        ts_array = src.read()
        print("Min Max Standardisation")
        print(Path(x).stem)
        print(f"len of unique {len(np.unique(ts_array))}")
        print("Shape")
        print(np.shape(ts_array))