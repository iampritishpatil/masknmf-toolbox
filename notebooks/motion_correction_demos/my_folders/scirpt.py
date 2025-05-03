# %%
import fastplotlib as fpl
import os
import sys
import masknmf
import tifffile
from pathlib import Path
import numpy as np
import torch
# %load_ext autoreload

# %%
data_path = Path(r"/Users/pritish/Developer/github/CaNaryG/data/1color/ImageData_Ch0_TP0000000.npy")
data = np.load(data_path,mmap_mode="r").astype(np.float32)

# %% [markdown]
# # If you don't have a good template estimate, run the generic template estimation procedure. End result is a PiecewiseRigidRegistrationStrategy object, used to register all frames to a template

# %%
rigid_strategy = masknmf.RigidMotionCorrection(max_shifts = [5, 5])
pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrection(strides = [24, 24],
                                                          overlaps = [8, 8],
                                                          max_rigid_shifts = [5, 5],
                                                          max_deviation_rigid = [2, 2])

pwrigid_strategy = masknmf.motion_correction.compute_template(data,
                                                              rigid_strategy,
                                                              num_iterations_piecewise_rigid = 1,
                                                              pwrigid_strategy = pwrigid_strategy,
                                                              device = "cpu",
                                                              batch_size = 100)

# %% [markdown]
# # Define a RegistrationArray that lazily loads motion corrected frames of the raw data
#%%
kernel = masknmf.motion_correction.gaussian_kernel(kernel_size=9, sigma=1.5,).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 7, 7]
temp_data = torch.from_numpy(data[0:10]).float().unsqueeze(1)  # Shape: [10, 1, 230, 803]

# Perform 2D convolution
conv = torch.nn.functional.conv2d(temp_data, kernel, padding='same') 
def filt_func(x):
    return torch.nn.functional.conv2d(x.float().unsqueeze(1), kernel, padding='same').squeeze(1)
# conv=torch.nn.functional.conv2d(temp_data, kernel)

#%%
data2 = masknmf.FilteredArray(data, filt_func, device = "cpu")
# %%
moco_results = masknmf.RegistrationArray(data2, pwrigid_strategy, device = "cpu",
                                        #  target_dataset=data,
                                         batch_size=100)

# %% [markdown]
# # Visualize with fastplotlib imagewidget
N = data.shape[0]

np.save("temp.npy", np.zeros(data.shape, dtype = np.uint16), allow_pickle=False)
out_data= np.load("temp.npy", mmap_mode="r+")
bs = 100 #batch size
for i in range(0, N, bs):
    temp= moco_results.index_frames_tensor(slice(i,i+bs)).cpu().numpy()
    out_data[slice(i,i+bs)] =temp.clip(0, 65535).astype(np.uint16)
    print(i)
    # break

#%%
# import numpy as np
# out_data= np.memmap("temp.npy", dtype = np.uint16, mode = "r+")

#%%
import napari 
#%%
viewer = napari.Viewer()
viewer.add_image(data2, name = "raw", colormap = "blue", blending="additive")
viewer.add_image(out_data, name = "moco", colormap = "green", blending="additive")
viewer.add_image(out_data.mean(axis=0), name = "mean", colormap = "red",blending="additive")

#%%
viewer.show()
# %%
# iw = fpl.ImageWidget(data = [data, moco_results])
# iw.cmap = "gray"
# iw.show()


# %%
