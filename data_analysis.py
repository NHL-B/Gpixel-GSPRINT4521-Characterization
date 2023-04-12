# %% Standard modules

import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
# from matplotlib_scalebar.scalebar import ScaleBar
import time
import pickle
import Gpixel as gp

# %% Functions

def roiAnalysis(folder_path, roi): 
    print('analyzing roi', roi)
    start = time.time() 
    gpixel = gp.GpixelAnalysis(roi, folder_path)
    ptc_mu, ptc_var, ptc_fit = gpixel.scan_gains()
    K, S = gpixel.get_sensitivity() # Conversion factor (DN'/e-), sensitivity (e-/DN)
    tdn = gpixel.get_TDN() # Temporal dark noise
    # fpn = gpixel.get_FPN() # Fixed-pattern noise 
    fwc = gpixel.get_FWC() # Full well capacity
    dr = gpixel.get_DR() # Dynamic range
    # dr_bits = gpixel.get_DR_bits()
    # snr = gpixel.get_SNR()
    res = {'K' : K, 'S' : S, 'tdn' : tdn, 'fwc' : fwc, 'dr' : dr}
    stop = time.time()
    print('Elapsed time...', round(stop-start, 1),'s')
    return res    

def roiGenerator(fov, n):
    roi_list = []
    steps = (fov[0]//n, fov[1]//n)
    for idx_row in range(n):
        for idx_col in range(n):
            r0, r1 = idx_row*steps[0], (idx_row+1)*steps[0]
            c0, c1 = idx_col*steps[1], (idx_col+1)*steps[1]
            roi = ((r0, c0), (r1, c1))
            roi_list.append(roi)
    return roi_list

def save_data(res, filename):
    with open(filename, 'wb') as f: # write binary
        pickle.dump(res, f)

def load_data(filename):
    with open(filename, 'rb') as f: # read binary
        data = pickle.load(f)
    return data

# %% Plots
def plot_PTC(poly_fit, x, y, gain='0', color='crimson'):
    gain = 'G{}'.format(gain)
    S = 1/poly_fit[0]
    x_fit = np.arange(0, 2**12)
    y_fit = x_fit*poly_fit[0]+poly_fit[1]
    plt.plot(x_fit, y_fit, '-k', lw=0.8)
    plt.plot(x, y, '-o', lw=0.4, color=color, label=gain + ': {:.2f} $e^-/ADU$'.format(S))
    plt.grid(which='major', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='-', alpha=0.2)
    # plt.xlim([0, 4200])
    plt.ylim([0, 2200]) # NOR12: 2200 / FBIN12: 350
    plt.xlim(left=0)
    plt.xlabel('$\mu_{light} - \mu_{dark0}$ [ADU]')
    plt.ylabel('$\sigma_{light}^2 - \sigma_{dark0}^2$ [$ADU^2 r.m.s.$]')
    plt.legend(loc='upper left', edgecolor="black")  
    
def plot_noise(x,y):
    plt.plot(x, y, linestyle='-', color='tab:blue', marker='o', lw=1.5)
    plt.grid(which='major', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='-', alpha=0.2)
    plt.xlim([0, 3]) 
    plt.xlabel('Gains')
    plt.title('Temporal dark noise [12NOR] - Gpixel GSPRINT4521', fontweight='bold')
    
def map_parameter(result, attribute='K', idx=0, n_points=2):
    par_m = [tmp[attribute][idx] for tmp in result]
    par_m = np.reshape(par_m, (n_points, n_points) )
    return par_m     

def hist_parameter(result, attribute='K', idx=0):
    par_h = [tmp[attribute][idx] for tmp in result]
    z = np.ravel(par_h) # multi-dimensional array into a contiguous flattened array
    return z

    
     # %%
if __name__ == '__main__':
    global fov
    fov = (4096, 5120) # NOR12: (4096, 5120) / FBIN12: (2016, 2560)
    roi = ((0, 0), fov) #  Default ROI (200x200)
    folder_path = './data_12NOR/' 
    g = ['G0', 'G1', 'G2', 'G3']
    
    # %% Data extraction
    gpixel = gp.GpixelAnalysis(roi, folder_path)
    ptc_mu, ptc_var, ptc_fit = gpixel.scan_gains()
    K, S = gpixel.get_sensitivity() # Conversion factor (DN'/e-), sensitivity (e-/DN)
    tdn = gpixel.get_TDN() # Temporal dark noise
    fpn = gpixel.get_FPN() # Fixed-pattern noise 
    fwc = gpixel.get_FWC() # Full well capacity
    dr = gpixel.get_DR() # Dynamic range
    dr_bits = gpixel.get_DR_bits()
    snr = gpixel.get_SNR() # Signal-to-noise ratio
    # prnu = gpixel.get_PRNU # Photo response non-uniformity
         
    n = 16 # Number of lines/columns, multiple of 2!!
    # roi_list = roiGenerator(fov, n)   
    # Data elaboration  
    # res = [roiAnalysis(folder_path, roi) for roi in roi_list]
    
    # Save data
    # save_data(res, 'data.pickle')
    
    # Load data
    loaded_data = load_data('data.pickle')
    
    # d = (fov[0]//n_points, fov[1]//n_points)
    # roi_list = [((ir*d[0], ic*d[1]), ((ir+1)*d[0], (ic+1)*d[1])) for ir in range(n_points) for ic in range(n_points)]
    
    # %% PTC plot
    plt.figure(dpi=500)
    colors = ['crimson', 'tab:blue', 'darkorange', 'darkgreen']
    for i in range(4):
        plot_PTC(ptc_fit[i], ptc_mu[i], ptc_var[i], gain=i, color=colors[i])
    plt.title('Photon Transfer Curve [12NOR] - Gpixel GSPRINT4521', fontweight='bold')
    plt.show()
    
    ## PTC [e-]
    plt.figure(dpi=500)
    for i in range(4):
        gain = f"G{i}"
        plt.plot(((ptc_mu[i]*gpixel.S[i])/1000), ((ptc_var[i]*gpixel.S[i]**2)/1000), lw=3, label=gain) # PTC (ADU) to PTC (e-), (ADU^2)*(e-/ADU)^2 => (e-)^2
    plt.grid(which='major', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='-', alpha=0.2)
    plt.xlim(left=0)
    plt.ylim(bottom=0) 
    plt.xlabel('$\mu_{light} - \mu_{dark0}$ [$ke^-$]')
    plt.ylabel('$\sigma_{light}^2 - \sigma_{dark0}^2$ [$(ke^-)^2r.m.s.$]')
    plt.legend(loc='upper left', edgecolor="black")
    plt.title('Photon Transfer Curve [12NOR] - Gpixel GSPRINT4521', fontweight='bold')
    plt.show()
    
    # %% SNR plot
    gain_values=['0', '1', '2', '3']
    idx = [np.where(gpixel.ptc_mu[i, :]>4000)[0][0] for i in np.int32(gain_values)]
    n_gains = len(g)
    plt.figure(dpi=500)
    [plt.loglog(gpixel.ptc_mu[i, :idx[i]]*gpixel.S[i], snr[i], '--s', markersize=3, label='G{}'.format(i)) for i in range(n_gains)]
    n_e = np.arange(10, 35000)
    plt.loglog(n_e, np.sqrt(n_e), '--k', label='Poisson')
    plt.grid(which='major', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='-', alpha=0.2)
    plt.legend(loc='upper left', edgecolor="black")
    plt.xlabel('Number of photo-electrons [$e^-$]')
    plt.ylabel('SNR []')
    plt.title('SNR [12NOR] - Gpixel GSPRINT4521', fontweight='bold')
    plt.show()

    # %% Noise plots
    plt.figure(dpi=500)
    for i in range(4):
        plot_noise(g, gpixel.tdn)
    plt.ylabel('$\sigma_{R}$ [$e^-$]')
    plt.show()
    
    plt.figure(dpi=500)
    for i in range(4):
        plot_noise(g, gpixel.tdn*gpixel.K)
    plt.ylabel('$\sigma_{R}$ [$ADU$]')
    plt.show()
    
    # %% ROI plots/Heatmap & Histogram
    gn = 2  # [0;3]
    item = 'S'
    
    par_m = map_parameter(loaded_data, attribute=item, idx=gn, n_points=n) # or res
    err_m = (par_m-np.mean(par_m))/np.mean(par_m)*100
    plt.figure(dpi=500)
    plt.imshow(par_m, extent=[0, n, 0, n], cmap='hot') # err or par
    cbar = plt.colorbar()
    cbar.set_label('[$e^-/ADU$]')
    plt.title('Sensitivity [G{}] - Gpixel GSPRINT4521'.format(gn))
    plt.show()

    par_h = hist_parameter(loaded_data, attribute=item, idx=gn)
    err_h = (par_h-np.mean(par_h))/np.mean(par_h)*100
    plt.figure(dpi=500)
    plt.hist(par_h, bins=np.arange(3.8, 4.4, 0.01), alpha=0.9)
    plt.grid(which='major', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='-', alpha=0.2)
    plt.xlabel('[$e^-/ADU$]')
    plt.ylabel('Counts')
    mean_item = round(np.mean(par_h), 2)
    plt.title(f'Mean_S [G{gn}]: {mean_item}e-/ADU / {par_m.shape} - GSPRINT4521')
    plt.axvline(mean_item, color='k', linestyle='dashed', linewidth=1.5, label='mean') 
    plt.legend(loc='upper left', edgecolor="black")
    # plt.title('${{\mu}}: {:.2f}e^-. {{\sigma}}: {:.2f}e^- r.m.s $'.format(np.mean(par_h), np.std(par_h)))
    plt.show()        

# %% Prints console 
print('\n')
for i in range(len(g)):
    print(f"• Temporal dark noise G{i}: {np.round(gpixel.tdn[i], decimals=1)} e-")
print('\n')
for i in range(len(g)):
    print(f'• Conversion factor G{i}: {np.round(gpixel.K[i], decimals=3)} ADU/e- (Sensitivity: {np.round(gpixel.S[i], decimals=3)} e-/ADU)')
print('\n')
for i in range(len(g)):
    print(f'• Full well capacity G{i}: {np.round(gpixel.fwc[i])} e-')
print('\n')
# for i in range(len(g)):
#     print(f'• Max SNR G{i}: {np.round(gpixel.snr[i], decimals=1)} dB')
# print('\n')   
for i in range(len(g)):
    print(f'• Dynamic range G{i}: {np.round(gpixel.dr[i], decimals=1)} dB ({np.round(gpixel.dr_bits[i], decimals=1)} bit)')
print('\n')
for i in range(len(g)):
    print(f'• FPN G{i}: {np.round(gpixel.fpn[i], decimals=1)} e-') 

# %% Plot Image  
# img = gpixel.get_light_stack()
# plt.figure(dpi=500)
# plt.imshow(img[50], cmap='gray', vmin=0, vmax=2**12)
# cbar = plt.colorbar()
# cbar.set_label('Mean (ADU)')
# plt.xlabel('x-axis (pixels)')
# plt.ylabel('y-axis (pixels)')
# scalebar = ScaleBar(4.5, 'um', location='lower left', box_alpha= 0.95, pad= 0.25, border_pad= 0.25, scale_loc='top', sep=3, length_fraction=0.25) # 1 pixel = 4.5 μm
# plt.gca().add_artist(scalebar)
# plt.show() 
