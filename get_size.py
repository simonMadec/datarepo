import os

import glob
from pathlib import Path

size = 0
for x_ in glob.glob(str(Path("data") / "data_reunion" ) + "/*/Training/*npy"):
    print(x_)
    size += os.path.getsize(x_)*0.000000001

breakpoint()
size = 0
count=0
for x_ in glob.glob(str(Path("data") / "data_reunion_origin_out150cmGSD_v2" ) + "/Sentinel-1/Training/*npy"):
    print(x_)
    size += os.path.getsize(x_)*0.000000001
    count=count+1

breakpoint()

size = 0
count=0
for x_ in glob.glob(str(Path("data") / "data_reunion_origin_out150cmGSD_v2" ) + "/Sentinel-2/Training/*npy"):
    print(x_)
    size += os.path.getsize(x_)*0.000000001
    count=count+1

breakpoint()
sizeP = 0
countP=0
for x_ in glob.glob(str(Path("data") / "data_dordogne_origin_out150cmGSD_v2" ) + "/Spot-P/Training/*npy"):
    print(x_)
    sizeP += os.path.getsize(x_)*0.000000001 
    countP=countP+1
     
breakpoint()
for x_ in glob.glob(str(Path("data") / "data_dordogne_origin_out150cmGSD_v2" ) + "/*/Training/*npy"):
    print(x_)
    size += os.path.getsize(x_) 