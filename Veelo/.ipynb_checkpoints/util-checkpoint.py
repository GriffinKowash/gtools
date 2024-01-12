# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:35:13 2023

@author: griffin.kowash
"""

#import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gtools import gtools
    

class Processing:
    @staticmethod
    def load_probe(filepath):
        
        """
        file = open(filepath, 'r')
        lines = file.readlines()
        file.close()
        
        time = []
        data = []
        timestep = []
        new_timestep = True
        
        for line in lines:
            #line_split = re.split('  | ', line) #is there a better way to split on generic white space?
            line_split = line.split()
            #line_split = line_split[1:]  #remove leading gap
            #line_split[-1] = line_split[-1][:-1]  #remove trailing newline character on last entry
            #print(line_split)
            
            try:
                values = list(map(lambda x: float(x), line_split))
            except:
                print(line_split)
            
            if new_timestep:  #could instead just check len(timestep)
                time.append(values[0])
                new_timestep = False
                timestep = timestep + values[1:]
                
            else:
                timestep = timestep + values
            
            if len(values) != 9:
                #could probably also use np.reshape
                x = timestep[::3]
                y = timestep[1::3]
                z = timestep[2::3]
                data.append([x, y, z])
                
                timestep = []
                new_timestep = True
        """
        
        time, data = gtools.load_distributed_probe(filepath, last_index='probe', precision='single')
                
        return np.array(time), np.array(data)
    
    @staticmethod
    def load_probes(filepaths):
        #loads multiple probe files and combines into one dataset
        data_sets = []
        
        for filepath in filepaths:
            t, d = Processing.load_probe(filepath)
            data_sets.append(d)
            
        data = np.concatenate(data_sets, axis=2)
        
        return t, data
    
    @staticmethod
    def load_source(source_name_and_path):
        #loads source time series
        t, source = np.loadtxt(source_name_and_path).T
        return t, source

    @staticmethod
    def load_source_fft(source_filepath):
        #loads plane wave source and return frequencies and fft
        source_data = np.loadtxt(source_filepath)
        dt = np.mean(source_data[1:, 0] - source_data[:-1, 0])
        source_freq = np.fft.rfftfreq(source_data.shape[0]) / dt
        source_fft = np.fft.rfft(source_data[:, 1], norm='forward') * 2
        
        return source_freq, source_fft

    @staticmethod
    def calc_magnitude(data):
        #takes in efield data in the form of load_probe or calc_fft results, returns vector magnitude
        return np.sqrt(np.abs(data[:, 0, :])**2 + np.abs(data[:, 1, :])**2 + np.abs(data[:, 2, :])**2)
    
    @staticmethod
    def calc_fft(t, data):
        #takes in time and efield components of the form of load_probe results, returns ex/ey/ex FFTs at each probe location       
        dt = np.mean(t[1:] - t[:-1])
        freq = np.fft.rfftfreq(t.size) / dt
        
        data_fft = np.fft.rfft(data, axis=0, norm='forward') * 2
        
        return freq, data_fft
    
    @staticmethod
    def calc_statistics(data):
        #takes in data (time, probes) and returns min, max, and mean over time between all probes
        data_min = data.min(axis=1)
        data_mean = data.mean(axis=1)
        data_max = data.max(axis=1)
        return data_min, data_mean, data_max

    @staticmethod
    def calc_shielding(data_fft, source_fft):
        #takes in data from calc_fft and plane wave fft to calcalate shielding effectiveness
        #assumes same time step and sample size for source and probe
        shielding = 20 * np.log10(np.abs(source_fft[:, np.newaxis] / data_fft))
        
        return shielding
    
    
class Workflows:
    @staticmethod
    def generate_shielding_statistics_from_results(probe_path, source_path):
        #takes in file paths, calculate shielding effectivess, and saves in probe_path directory
        if type(probe_path) == list or type(probe_path) == tuple:
            #stacks results from multiple probes
            t, data = Processing.load_probes(probe_path)
        else:
            #processes single probe file
            t, data = Processing.load_probe(probe_path)
        
        freq, e_fft = Processing.calc_fft(t, data)
        e_fft_mag = Processing.calc_magnitude(e_fft)
        
        #_, source_fft = Processing.load_source_fft(source_path)
        _, esource = Processing.load_source(source_path)
        esource = gtools.pad_array_to_length(esource, t.size, 0)
        _, source_fft = Processing.calc_fft(t, esource)
                
        se = Processing.calc_shielding(e_fft_mag, source_fft)
        se_min, se_mean, se_max = Processing.calc_statistics(se)
        
        path = '\\'.join(source_path.split('\\')[:-1] + ['se_stats.dat'])
        np.savetxt(path, np.transpose([freq, se_min, se_mean, se_max]))
        
        print('Shielding statistics saved to ', path, '\n')
        
        return freq, se_min, se_mean, se_max
        
    
    @staticmethod
    def plot_shielding_statistics_from_file(se_path, title=None, ylim=None):
        #takes in filepath and generates plot of min, max, and mean shielding effectiveness
        freq, se_min, se_mean, se_max = np.loadtxt(se_path).T
    
        sns.set()
        plt.fill_between(freq, se_min, se_max, color=(0.5, 0.5, 0.5, 0.5))
        plt.plot(freq, se_mean, linestyle='--', color='C0')
        
        plt.xlim(1e7, 5e10)
        plt.xscale('log')
        
        if ylim == None:
            plt.ylim(15, 150)
        else:
            plt.ylim(ylim[0], ylim[1])
        
        if title != None:
            plt.suptitle(title)
            
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Shielding (dB)')
        
        plt.show()
        
        
    @staticmethod
    def plot_shielding_statistics_from_multiple_runs(se_paths, title=None):
        se_mins = []
        se_means = []
        se_maxs = []
        
        for se_path in se_paths:
            freq, se_min, se_mean, se_max = np.loadtxt(se_path).T
            se_mins.append(se_min)
            se_means.append(se_mean)
            se_maxs.append(se_max)
            
        se_mins = np.array(se_mins)
        se_means = np.array(se_means)
        se_maxs = np.array(se_maxs)
        
        se_min_all = np.min(se_mins, axis=0)
        se_mean_all = np.mean(se_means, axis=0)
        se_max_all = np.max(se_maxs, axis=0)
        
        plt.fill_between(freq, se_min_all, se_max_all, color=(0.5, 0.5, 0.5, 0.5), label='Range')
        #plt.fill_between(freq, se_min_all, se_max_all, color=(.30, .45, .69, .5))    #(.87, .52, .32, .5))
        plt.plot(freq, se_mean_all, linestyle='--', color='C0', label='Mean')
        plt.xlim(1e7, 5e10)
        plt.ylim(0, 150)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Shielding effectiveness (dB)')
        
        if title == None:
            plt.suptitle('Shielding effectiveness, all runs')
        else:
            plt.suptitle(title)
            
        plt.legend()
        plt.xscale('log')
        plt.show()
    
    
    @staticmethod
    def plot_electric_field_time_domain(probe_paths, suffix=None):
        fig1, ax1 = plt.subplots(1)
        fig2, ax2 = plt.subplots(1)
                
        if type(probe_paths) in [list, tuple]:           
            t, data = Processing.load_probes(probe_paths)
        else:
            t, data = Processing.load_probe(probe_paths)
        
        # E-field magnitude
        emag = Processing.calc_magnitude(data)
        emag_min, emag_mean, emag_max = Processing.calc_statistics(emag)
        
        ax1.fill_between(t, emag_min, emag_max, color=(.5, .5, .5, .5), label='Range')
        ax1.plot(t, emag_mean, color='C0', label='Mean')
        ax1.legend()
        if suffix == None:
            fig1.suptitle('Electric field magnitude time series')
        else:
            fig1.suptitle(f'Electric field magnitude time series ({suffix})')
    
        # E-field components
        ex, ey, ez = data[:,0,:], data[:,1,:], data[:,2,:]
        _, ex_mean, _ = Processing.calc_statistics(ex)
        _, ey_mean, _ = Processing.calc_statistics(ey)
        _, ez_mean, _ = Processing.calc_statistics(ez)

        ax2.plot(t, ex_mean, label='Ex')
        ax2.plot(t, ey_mean, label='Ey')
        ax2.plot(t, ez_mean, label='Ez')
        
        ax2.legend()
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Electric field (V/m)')
        if suffix == None:
            fig2.suptitle('Electric field components time series')
        else:
            fig2.suptitle(f'Electric field components time series ({suffix})')

        # Display figures (comment out for IPython)
        #fig1.show()
        #fig2.show()
        
        
    @staticmethod
    def plot_with_range(x, ymin, ymean, ymax, xlabel, ylabel, title, xscale='linear', yscale='linear', xlim=None, ylim=None):
        fig, ax = plt.subplots(1)
        ax.fill_between(x, ymin, ymax, color=(.5, .5, .5, .5), label='Range')
        ax.plot(x, ymean, label='Mean', linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        fig.suptitle(title)
        #fig.show()
        
        
    @staticmethod
    def plot_cable_from_multiple_runs(Isc_paths, Voc_paths):    
        # refactor later. Plots Voc and Isc data with statistics across multuple sims
        
        t = None
        f = None
        
        i_c1_all = []
        i_c2_all = []
        i_c3_all = []
        i_shield_all = []
        
        v_c1_all = []
        v_c2_all = []
        v_c3_all = []
        v_shield_all = []
        
        i_c1_fft_all = []
        i_c2_fft_all = []
        i_c3_fft_all = []
        i_shield_fft_all = []
        
        v_c1_fft_all = []
        v_c2_fft_all = []
        v_c3_fft_all = []
        v_shield_fft_all = []
        
        
        for path in Isc_paths:
            # Load time series data
            _, i_c1 = np.loadtxt('\\'.join([path, 'Current_(C1).dat'])).T
            _, i_c2 = np.loadtxt('\\'.join([path, 'Current_(TSP_C2).dat'])).T
            _, i_c3 = np.loadtxt('\\'.join([path, 'Current_(TSP_C3).dat'])).T
            _, i_shield = np.loadtxt('\\'.join([path, 'Current_(TSP_shield).dat'])).T
            
            # Generate FFTs
            i_c1_fft = np.abs(np.fft.rfft(i_c1, norm='forward') * 2)
            i_c2_fft = np.abs(np.fft.rfft(i_c2, norm='forward') * 2)
            i_c3_fft = np.abs(np.fft.rfft(i_c3, norm='forward') * 2)
            i_shield_fft = np.abs(np.fft.rfft(i_shield, norm='forward') * 2)
            
            # Append data from run to full collection
            i_c1_all.append(i_c1)
            i_c2_all.append(i_c2)
            i_c3_all.append(i_c3)
            i_shield_all.append(i_shield)
            
            i_c1_fft_all.append(i_c1_fft)
            i_c2_fft_all.append(i_c2_fft)
            i_c3_fft_all.append(i_c3_fft)
            i_shield_fft_all.append(i_shield_fft)
            
            if t is None:
                t = _
                dt = t[1] - t[0]
                f = np.fft.rfftfreq(t.size, d=dt)
                
                
        for path in Voc_paths:
            _, v_c1 = np.loadtxt('\\'.join([path, 'Voltage_(C1_J1).dat'])).T
            _, v_c2 = np.loadtxt('\\'.join([path, 'Voltage_(TSP_C2_J1).dat'])).T
            _, v_c3 = np.loadtxt('\\'.join([path, 'Voltage_(TSP_C3_J1).dat'])).T
            _, v_shield = np.loadtxt('\\'.join([path, 'Voltage_(TSP_shield_J1).dat'])).T
            
            v_c1_fft = np.abs(np.fft.rfft(v_c1, norm='forward') * 2)
            v_c2_fft = np.abs(np.fft.rfft(v_c2, norm='forward') * 2)
            v_c3_fft = np.abs(np.fft.rfft(v_c3, norm='forward') * 2)
            v_shield_fft = np.abs(np.fft.rfft(v_shield, norm='forward') * 2)
            
            v_c1_all.append(v_c1)
            v_c2_all.append(v_c2)
            v_c3_all.append(v_c3)
            v_shield_all.append(v_shield)
            
            v_c1_fft_all.append(v_c1_fft)
            v_c2_fft_all.append(v_c2_fft)
            v_c3_fft_all.append(v_c3_fft)
            v_shield_fft_all.append(v_shield_fft)
            
            
        # Reshape full datasets to form (time, runs) to use with Processing.calc_statistics
        i_c1_all = np.array(i_c1_all).T
        i_c2_all = np.array(i_c2_all).T
        i_c3_all = np.array(i_c3_all).T
        i_shield_all = np.array(i_shield_all).T
        
        v_c1_all = np.array(v_c1_all).T
        v_c2_all = np.array(v_c2_all).T
        v_c3_all = np.array(v_c3_all).T
        v_shield_all = np.array(v_shield_all).T
        
        i_c1_fft_all = np.array(i_c1_fft_all).T
        i_c2_fft_all = np.array(i_c2_fft_all).T
        i_c3_fft_all = np.array(i_c3_fft_all).T
        i_shield_fft_all = np.array(i_shield_fft_all).T
        
        v_c1_fft_all = np.array(v_c1_fft_all).T
        v_c2_fft_all = np.array(v_c2_fft_all).T
        v_c3_fft_all = np.array(v_c3_fft_all).T
        v_shield_fft_all = np.array(v_shield_fft_all).T
        
        # Calculate min, mean, and max
        i_c1_min, i_c1_mean, i_c1_max = Processing.calc_statistics(i_c1_all)
        i_c2_min, i_c2_mean, i_c2_max = Processing.calc_statistics(i_c2_all)
        i_c3_min, i_c3_mean, i_c3_max = Processing.calc_statistics(i_c3_all)
        i_shield_min, i_shield_mean, i_shield_max = Processing.calc_statistics(i_shield_all)
        
        v_c1_min, v_c1_mean, v_c1_max = Processing.calc_statistics(v_c1_all)
        v_c2_min, v_c2_mean, v_c2_max = Processing.calc_statistics(v_c2_all)
        v_c3_min, v_c3_mean, v_c3_max = Processing.calc_statistics(v_c3_all)
        v_shield_min, v_shield_mean, v_shield_max = Processing.calc_statistics(v_shield_all)
        
        i_c1_fft_min, i_c1_fft_mean, i_c1_fft_max = Processing.calc_statistics(i_c1_fft_all)
        i_c2_fft_min, i_c2_fft_mean, i_c2_fft_max = Processing.calc_statistics(i_c2_fft_all)
        i_c3_fft_min, i_c3_fft_mean, i_c3_fft_max = Processing.calc_statistics(i_c3_fft_all)
        i_shield_fft_min, i_shield_fft_mean, i_shield_fft_max = Processing.calc_statistics(i_shield_fft_all)
        
        v_c1_fft_min, v_c1_fft_mean, v_c1_fft_max = Processing.calc_statistics(v_c1_fft_all)
        v_c2_fft_min, v_c2_fft_mean, v_c2_fft_max = Processing.calc_statistics(v_c2_fft_all)
        v_c3_fft_min, v_c3_fft_mean, v_c3_fft_max = Processing.calc_statistics(v_c3_fft_all)
        v_shield_fft_min, v_shield_fft_mean, v_shield_fft_max = Processing.calc_statistics(v_shield_fft_all)
            
        # Generate plots
        #fig, ax = plt.subplots(1)
        #ax.fill_between(t, i_c1_min, i_c1_max, color=(.5, .5, .5, .5), label='Range')
        #ax.plot(t, i_c1_mean, label='Mean', linestyle='--')
        #ax.set_xlabel('Time (s)')
        #ax.set_ylabel('Current (A)')
        #ax.legend()
        #fig.suptitle('Bare conductor - current (time series)')
        #fig.show()
        
        Workflows.plot_with_range(t, i_c1_min, i_c1_mean, i_c1_max, xlabel='Time (s)', ylabel='Current (A)', title='Bare conductor - Isc (time series)', ylim=(-0.05, 0.05))
        Workflows.plot_with_range(t, i_c2_min, i_c2_mean, i_c2_max, xlabel='Time (s)', ylabel='Current (A)', title='TSP conductor 1 - Isc (time series)', ylim=(-0.05, 0.05))
        Workflows.plot_with_range(t, i_c3_min, i_c3_mean, i_c3_max, xlabel='Time (s)', ylabel='Current (A)', title='TSP conductor 2 - Isc (time series)', ylim=(-0.05, 0.05))
        Workflows.plot_with_range(t, i_shield_min, i_shield_mean, i_shield_max, xlabel='Time (s)', ylabel='Current (A)', title='TSP shield - Isc (time series)', ylim=(-0.05, 0.05))

        Workflows.plot_with_range(f, i_c1_fft_min, i_c1_fft_mean, i_c1_fft_max, xlabel='Frequency (Hz)', ylabel='Current (A)', title='Bare conductor - Isc (FFT)', xscale='log', yscale='log', xlim=(1e7, 5e10), ylim=(1e-8, 2e-2))
        Workflows.plot_with_range(f, i_c2_fft_min, i_c2_fft_mean, i_c2_fft_max, xlabel='Frequency (Hz)', ylabel='Current (A)', title='TSP conductor 1 - Isc (FFT)', xscale='log', yscale='log', xlim=(1e7, 5e10), ylim=(1e-8, 2e-2))
        Workflows.plot_with_range(f, i_c3_fft_min, i_c3_fft_mean, i_c3_fft_max, xlabel='Frequency (Hz)', ylabel='Current (A)', title='TSP conductor 2 - Isc (FFT)', xscale='log', yscale='log', xlim=(1e7, 5e10), ylim=(1e-8, 2e-2))
        Workflows.plot_with_range(f, i_shield_fft_min, i_shield_fft_mean, i_shield_fft_max, xlabel='Frequency (Hz)', ylabel='Current (A)', title='TSP shield - Isc (FFT)', xscale='log', yscale='log', xlim=(1e7, 5e10), ylim=(1e-8, 2e-2))

        Workflows.plot_with_range(t, v_c1_min, v_c1_mean, v_c1_max, xlabel='Time (s)', ylabel='Voltage (V)', title='Bare conductor - Voc (time series)', ylim=(-3.2, 3.2))
        Workflows.plot_with_range(t, v_c2_min, v_c2_mean, v_c2_max, xlabel='Time (s)', ylabel='Voltage (V)', title='TSP conductor 1 - Voc (time series)', ylim=(-3.2, 3.2))
        Workflows.plot_with_range(t, v_c3_min, v_c3_mean, v_c3_max, xlabel='Time (s)', ylabel='Voltage (V)', title='TSP conductor 2 - Voc (time series)', ylim=(-3.2, 3.2))
        Workflows.plot_with_range(t, v_shield_min, v_shield_mean, v_shield_max, xlabel='Time (s)', ylabel='Voltage (V)', title='TSP shield - Voc (time series)', ylim=(-3.2, 3.2))

        Workflows.plot_with_range(f, v_c1_fft_min, v_c1_fft_mean, v_c1_fft_max, xlabel='Frequency (Hz)', ylabel='Voltage (V)', title='Bare conductor - Voc (FFT)', xscale='log', yscale='log', xlim=(1e7, 5e10), ylim=(1e-7, 2e0))
        Workflows.plot_with_range(f, v_c2_fft_min, v_c2_fft_mean, v_c2_fft_max, xlabel='Frequency (Hz)', ylabel='Voltage (V)', title='TSP conductor 1 - Voc (FFT)', xscale='log', yscale='log', xlim=(1e7, 5e10), ylim=(1e-7, 2e0))
        Workflows.plot_with_range(f, v_c3_fft_min, v_c3_fft_mean, v_c3_fft_max, xlabel='Frequency (Hz)', ylabel='Voltage (V)', title='TSP conductor 2 - Voc (FFT)', xscale='log', yscale='log', xlim=(1e7, 5e10), ylim=(1e-7, 2e0))
        Workflows.plot_with_range(f, v_shield_fft_min, v_shield_fft_mean, v_shield_fft_max, xlabel='Frequency (Hz)', ylabel='Voltage (V)', title='TSP shield - Voc (FFT)', xscale='log', yscale='log', xlim=(1e7, 5e10), ylim=(1e-7, 2e0))
