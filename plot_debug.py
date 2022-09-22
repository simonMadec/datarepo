

import glob
import numpy as np
import matplotlib.pyplot as plt 

sensor = "Spot-P"
ms = "Spot-MS"
s2 = "Sentinel-2"

site = "data_dordogne"
site = "data_ToyDordogne_out150cmGSD"
split = 0

for pathin in ['Training']:
    path_Y_train = np.load(f"/home/simon/DATA/land_use_classification/{site}/Ground_truth/{pathin}/Ground_truth_{pathin}_split_{split}.npy")
    path_X_train_Spot = np.load(f"/home/simon/DATA/land_use_classification/{site}/{sensor}/{pathin}/{sensor}_{pathin}_split_{split}.npy")
    path_X_train_MS = np.load(f"/home/simon/DATA/land_use_classification/{site}/{ms}/{pathin}/{ms}_{pathin}_split_{split}.npy")
    path_X_train_S2 = np.load(f"/home/simon/DATA/land_use_classification/{site}/{s2}/{pathin}/{s2}_{pathin}_split_{split}.npy")

    ii = 9

    paa = path_X_train_Spot[10,:,:,0]
    mss = path_X_train_MS[10,:,:,0:4]
    s2 =  path_X_train_S2[10,:,:,:]

    plt.close()
    mosaic = """
            AAAA
            AAAA
            AAAA
            AAAA
            BCDE"""

    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(mosaic,subplot_kw={'xticks': [], 'yticks': []})
    A = ax_dict["A"].imshow(paa.astype("float32"),cmap='binary')
    plt.colorbar(A,ax=ax_dict["A"])
    B = ax_dict["B"].imshow(mss[:,:,0].astype("float32"),cmap='binary')
    ax_dict["C"].imshow(mss[:,:,1].astype("float32"),cmap='binary')
    ax_dict["D"].imshow(mss[:,:,2].astype("float32"),cmap='binary')
    E = ax_dict["E"].imshow(mss[:,:,3].astype("float32"),cmap='binary')

    plt.colorbar(E,ax=ax_dict["E"])
    plt.colorbar(E,ax=ax_dict["E"])
    plt.savefig(f"check1.png")

    breakpoint()
    plt.close()
    mosaic = """
            AAAA
            AAAA
            AAAA
            AAAA
            BCDE
            FGHI"""

    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(mosaic,subplot_kw={'xticks': [], 'yticks': []})
    A = ax_dict["A"].imshow(paa.astype("float32"),cmap='binary')
    plt.colorbar(A,ax=ax_dict["A"])
    B = ax_dict["B"].imshow(s2[:,:,0].astype("float32"),cmap='binary')
    ax_dict["C"].imshow(s2[:,:,1].astype("float32"),cmap='binary')
    ax_dict["D"].imshow(s2[:,:,2].astype("float32"),cmap='binary')
    E = ax_dict["E"].imshow(s2[:,:,3].astype("float32"),cmap='binary')

    B = ax_dict["F"].imshow(s2[:,:,4].astype("float32"),cmap='binary')
    ax_dict["G"].imshow(s2[:,:,5].astype("float32"),cmap='binary')
    ax_dict["H"].imshow(s2[:,:,6].astype("float32"),cmap='binary')
    E = ax_dict["I"].imshow(s2[:,:,7].astype("float32"),cmap='binary')

    plt.colorbar(E,ax=ax_dict["E"])
    plt.colorbar(E,ax=ax_dict["E"])
    plt.savefig(f"check3.png")

    # plot = 1
    # if plot:
    #     for i in range(0,paa.shape[0],2000):
            
    #         paai = paa[i,:,:,0].reshape([1,32,32])

    #         mssi = mss[i,:,:,:].transpose((-1,0,1))

