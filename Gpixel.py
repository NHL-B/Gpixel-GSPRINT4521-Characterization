import numpy as np
import os
import tifffile as tiff

class GpixelAnalysis:
    def __init__(self, roi, folder_path):
        self.roi = roi
        self.folder_path = folder_path
    
    def get_file_list(self, path):
        '''List all the content of the selected folder'''
        return os.listdir(path)

    def load_data(self, data_path):
        '''Loading the .tiff image'''
        frame = tiff.imread(data_path) 
        return frame 
    
    def apply_roi(self, frame, roi): 
        '''Apply a region of interest to the data'''
        r1, c1 = roi[0] # Specify top-left corner
        r2, c2 = roi[1] # Specify bottom-right corner
        frame = frame[r1:r2, c1:c2]
        return frame

    def get_frame_stack(self, folder_path, file_list, roi):
        """Class to load a full stack of pictures, applying a ROI"""
        n_points = len(file_list)
        size = n_points, roi[1][0]-roi[0][0], roi[1][1]-roi[0][1]
        frame_stack = np.zeros((size), np.int16) # Integer (-32768 to 32767)
        for i in range(n_points):
            data_path = folder_path + file_list[i]
            frame = self.load_data(data_path)
            frame_stack[i, :] = self.apply_roi(frame, roi)
        return frame_stack    
    
    def get_light_stack(self, gain_value='0'):
        path = self.folder_path + r'G{}/'.format(gain_value)
        file_list = self.get_file_list(path)
        light_stack = self.get_frame_stack(path, file_list, self.roi)
        return light_stack
    
    # z = gpixel.get_light_stack(gain_value=0)
    # print(z)
    # z0 = z[0,:,:].reshape(1,200,200)
    # print(z0)
    # np.mean(z0)  
    # np.std(z0)
    
    def get_dark_stack(self, gain_value='0', offset=128):
        path = self.folder_path + r'D{}/'.format(gain_value)
        file_list = self.get_file_list(path)
        dark_stack = self.get_frame_stack(path, file_list, self.roi)
        dark_stack -= offset
        return dark_stack
    
    def get_stack_stats(self, frame_stack):
        n_points = frame_stack.shape[0]
        mu = [np.mean((frame_stack[i, :, :] + frame_stack[i+1, :, :]))/2 for i in range(0, n_points, 2)]
        var = [np.var((frame_stack[i, :, :] - frame_stack[i+1, :, :]))/2 for i in range(0, n_points, 2)] 
        return np.array(mu), np.array(var) 

    def get_PTC(self, light_stack, dark_stack=0):
        '''Get mean (mu) and variance (var) to build PTC curve'''
        mu_d, var_d = self.get_stack_stats(dark_stack)
        mu_l, var_l = self.get_stack_stats(light_stack)
        noise_l =  np.sqrt(var_l)
        noise_d = np.sqrt(var_d)
        ptc_mu = mu_l - mu_d 
        ptc_var = var_l - var_d
        return ptc_mu, ptc_var, noise_l, noise_d 
    
    def fit_PTC(self, x, y, th_value=0.7): # TODO
        adu_th = (np.nanmax(y) - y[0])*th_value + y[0]
        idx = np.where(y>adu_th)[0][0]
        poly_fit = np.polyfit(x[:idx], y[:idx], 1)
        return poly_fit
    
    def scan_gains(self, gain_values=['0', '1', '2', '3']):
        ptc_mu = []
        ptc_var = []
        ptc_fit = []
        noise_l = []
        noise_d = []
        
        for i, gain in enumerate(gain_values):    
            light_stack = self.get_light_stack(gain_value=gain)
            dark_stack = self.get_dark_stack(gain_value=gain)
            mu, var, std_l, std_d = self.get_PTC(light_stack, dark_stack)
            ptc_mu.append(mu)
            ptc_var.append(var) 
            noise_l.append(std_l)
            noise_d.append(std_d)
            ptc_fit.append(self.fit_PTC(mu, var))
        self.ptc_mu = np.array(ptc_mu)
        self.ptc_var = np.array(ptc_var)
        self.noise_l = np.array(noise_l)
        self.noise_d = np.array(noise_d)
        self.ptc_fit = np.array(ptc_fit)
        return ptc_mu, ptc_var, ptc_fit
    
    def get_sensitivity(self):
        n = self.ptc_fit.shape[0]
        K = np.array([self.ptc_fit[i][0] for i in range(n)])
        S = np.array([1/self.ptc_fit[i][0] for i in range(n)])
        self.S = S # e-/ADU 
        self.K = K # ADU/e-
        return K, S
    
    def get_TDN(self):   
        # self.get_sensitivity() # TODO !
        n = self.noise_d.shape[0]
        tdn = np.array([self.noise_d[i]*self.S[i] for i in range(n)]) # e-
        tdn = np.reshape(tdn, n)
        self.tdn = tdn
        return tdn
     
    def get_FPN(self, gain_values=['0', '1', '2', '3']):    
        var_tot = [] # Variance of the dark frame (fixed noise + temporal dark noise in ADU)
        var_tdn = [] # Temporal dark noise (ADU)
        for i, gain in enumerate(gain_values):    
            dark_stack = self.get_dark_stack(gain_value=gain)
            var1 = (np.var(dark_stack[0, :, :]) + np.var(dark_stack[0, :, :]))/2
            var2 = np.var(dark_stack[0, :, :] - dark_stack[1, :, :])/2
            var_tot.append(var1)
            var_tdn.append(var2)
        var_fpn = np.array(var_tot)-np.array(var_tdn)
        fpn = np.sqrt(var_fpn)*self.S
        n = fpn.shape[0]
        self.fpn = np.reshape(fpn, n)
        return fpn 
    
    def get_FWC(self):
        idx_max = np.argmax(self.ptc_var, axis=1)
        mu_max = [self.ptc_mu[i][idx_max[i]] for i in range(4)]
        fwc = mu_max*self.S # or /self.K
        self.fwc = fwc
        return fwc
    
    def get_SNR(self, gain_values=['0', '1', '2', '3']):  # Square root of the FWC (e-), the unit of dB is 20 x log (SNR). # [20*np.log10(np.sqrt(fwc[i])) for i in range(4)]
        # idx_max = np.argmax(self.ptc_var, axis=1)
        # mu_max = [self.ptc_mu[i][idx_max[i]] for i in range(4)]
        # noise_l_max = [self.noise_l[i][idx_max[i]] for i in range(4)]
        # snr = [20*np.log10((mu_max[i])/(noise_l_max[i])) for i in range(4)] 
        idx = [np.where(self.ptc_mu[i, :]>4000)[0][0] for i in np.int32(gain_values)]
        n_gains = len(gain_values)
        snr = [self.ptc_mu[i, :idx[i]]/self.noise_l[i, :idx[i]] for i in range(n_gains)]
        self.snr = snr 
        return snr
    
    def get_DR(self):
        lst = [20*np.log10((self.fwc[i])/(self.tdn[i])) for i in range(4)]
        dr = np.array(lst)
        self.dr = dr 
        return dr
    
    def get_DR_bits(self):
        dr_bits = [np.log10((self.fwc[i])/(self.tdn[i]))/(np.log10(2)) for i in range(4)]
        self.dr_bits = dr_bits
        return dr_bits
