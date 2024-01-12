import warnings
import time
import os
import glob

import numpy as np


def rfft(t, x):
    """Calculates FFT from real time series data.
    
    The result is normalized such that the FFT provides the true amplitude of each frequency.
    
    The data array (x) is expected to have time steps along the first axis. Multiple data sets
    can be processed simultaneously by stacking along additional axes.
    
    Parameters
    ----------
    t : np.ndarray
        Time step data (1d)
    x : np.ndarray
        Time series data (nd)

    Returns
    -------
    tuple
        A tuple of ndarrays of the form (frequency, FFT)
    """
    
    # Handle errors and warnings
    if t.ndim > 1:
        raise ValueError(f'Array t must have exactly one dimension; {t.ndim} provided.')
        
    elif t.size != x.shape[-1]:
        raise ValueError(f'Last dimension of x ({x.shape[-1]}) must match size of t ({t.size}).')
        
    elif np.any(np.iscomplexobj(x)):
        warnings.warn(f'Array x has complex dtype {x.dtype}; imaginary components will be discarded, which may affect results.')
    
    # Compute FFT and frequency array
    f = np.fft.rfftfreq(t.size) / (t[1] - t[0])
    x_fft = np.fft.rfft(x, norm='forward', axis=-1) * 2
    
    return f, x_fft


def shielding(t, x, xref):
    """Calculates shielding effectiveness from time series data.
    
    The result is normalized such that the FFT provides the true amplitude of each frequency.
    
    The data array (x) is expected to have time steps along the first axis. Multiple data sets
    can be processed simultaneously by stacking along additional axes.
    
    Parameters
    ----------
    t : np.ndarray
        Time step data (1d)
    x : np.ndarray
        Time series data (nd)
    xref: np.ndarray
        Time series data for reference waveform (nd)

    Returns
    -------
    tuple : ndarray, ndarray
        A tuple of the form (frequency, FFT)
    """
    
    # Handle errors and warnings
    if t.ndim > 1:
        raise ValueError(f'Array t must have exactly one dimension; {t.ndim} provided.')
        
    elif x.shape[-1] != t.size:
        raise ValueError(f'Last dimension of x ({x.shape[-1]}) must match size of t ({t.size}).')
        
    elif np.any(xref.shape != x.shape):
        raise ValueError(f'Shape of xref ({xref.shape}) must match x ({x.shape}).')
        
    elif np.any(np.iscomplex(x)):
        warnings.warn(f'Array x has complex dtype {x.dtype}; imaginary components will be disregarded, which may affect results.')
    
    # Compute FFTs and frequency array
    f = np.fft.rfftfreq(t.size) / (t[1] - t[0])
    x_fft = np.fft.rfft(x, norm='forward', axis=0) * 2
    xref_fft = np.fft.rfft(xref, norm='forward', axis=0) * 2
    
    # Compute shielding (dB)
    se = 20 * np.log10(np.abs(xref_fft / x_fft))
    
    return f, se


def trim_to_time(t, x, cutoff):
    """Trims time domain data to a specified cutoff time.
    
    Can be used on a single 
    
    Parameters
    ----------
    t : np.ndarray
        Time step data (1d)
    x : np.ndarray
        Time series data (nd)
    cutoff : float
        Cutoff time in seconds

    Returns
    -------
    tuple : ndarray, ndarray
        A tuple of trimmed data the form (t_trim, x_trim)
    """
    
    # Handle errors and warnings
    if t.ndim > 1:
        raise ValueError(f'Array t must have exactly one dimension; {t.ndim} provided.')
        
    elif x.shape[0] != t.size:
        raise ValueError(f'First dimension of x ({x.shape[0]}) must match size of t ({t.size}).')
        
    elif cutoff < 0:
        raise ValueError(f'Cutoff time ({cutoff}) must be greater than or equal to zero.')
        
    # Identify cutoff index and return trimmed data
    index = np.abs(t - cutoff).argmin()
    t_trim = t[:index]
    x_trim = x[:index, ...]
    
    return t_trim, x_trim


def restrict_surface_current(path, direction, overwrite=True):
    """Restricts surface current definition in emin file to a single direction.
    
    As of 2024R1, the direction of a surface current cannot be specified in the GUI.
    For example, a current source applied to a z-normal surface will have
    currents in both the x and y directions. This function can modify such a
    current source to be directed only in the x or y direction.
    
    Parameters
    ----------
    path : str
        Path to emin file or directory containing emin file
    direction : int | str
        Desired current direction (0|'x', 1|'y', 2|'z')
    overwrite : bool (option)
        Whether to overwrite emin or save timestamped copy

    Returns
    -------
    None
    """
    
    # Map between direction string and column index
    column_dict = {'x': 3, 'y': 4, 'z': 5}
    
    # Handle exceptions and warnings
    if direction not in column_dict.keys():
        raise ValueError(f'Direction must be "x", "y", or "z" (provided "{direction}")')
        
    # open emin file
    emin_path_and_name = find_emin(path)
    emin = open(emin_path_and_name, 'r')
    lines = emin.readlines()
    emin.close()

    # identify start and end of current source definition
    i0, i1 = 0, 0

    for i, line in enumerate(lines):
        # TODO: evaluate flexibility of this approach
        if 'SourceCurrent.dat' in line:
            i0 = i + 1

        elif 'PROBES' in line:
            i1 = i - 2
            break

    # Only retain lines with non-zero values in the desired column
    column = column_dict[direction]
    probe = [line for line in lines[i0:i1] if float(line.split()[column]) != 0]
    new_lines = lines[:i0] + probe + lines[i1:]
    
    if len(probe) == 0:
        warnings.warn(f'No {direction}-directed source elements found; probe definition deleted.')

    # Save emin
    if overwrite:
        save_path_and_name = emin_path_and_name
    else:
        save_path_and_name = emin_path_and_name[:-5] + '_' + str(time.time()) + '.emin'
    
    emin = open(save_path_and_name, 'w')
    emin.writelines(new_lines)
    emin.close()

    print(f'Saved modified emin file to {save_path_and_name}')
        
    
def find_emin(path):
    """Helper function to identify an emin file within a directory.
    
    The "path" argument can also point directly to the emin file rather
    than the containing directory to improve flexibility for users.
    
    Parameters
    ----------
    path : str
        Path to emin file or directory containing emin file

    Returns
    -------
    str | None
        Full path and name to emin file, or None if absent.
    """
    
    # check for existence of file/directory
    if not os.path.exists(path):
        raise Exception(f'Path specified by user does not exist. ({path})')
    
    # determine emin path and name from "path" argument
    if path[-4:] == 'emin':
        path_and_name = path
    
    else:
        emins = glob.glob('\\'.join([path, '*.emin']))
        
        if len(emins) > 0:
            path_and_name = emins[0]
            if len(emins) > 1:
                warnings.warn(f'Multiple emin files found in directory; selecting {path_and_name}.')
                
        else:
            raise Exception(f'No emin file found in specified directory ({path})')
            
    return path_and_name


def load_distributed_probe(path_and_name, last_index='probe', precision='single'):
    """Converts distributed and box probe results into a numpy array.
    
    For readability, each time step of a distributed probe is split into
    multiple output lines with nine entries in each, with measured values
    for each point listed in order x, y, z. Since values must be written
    out in multiples of three, and the initial time step value introduces
    a one-entry offset, there will always be a line with fewer than nine
    entries at the end of each time step, allowing the file to be parsed.
    
    The dimensions of the returned data depend on the 'last_index' argument:
        'probe':     (time, component, probe)
        'component': (time, probe, component)
         None:       (time, probes * components)
    
    Parameters
    ----------
    path_and_name : str
        Path to probe file (with .dat suffix)
    last_index : str (optional)
        Specifies structure of data array by last index
    precision : str (optional)
        Precision of simulation results ('single' | 'double')

    Returns
    -------
    tuple : np.ndarray, np.ndarray
        A tuple of time steps and probe results in the form (t, data)
    """
    
    # Handle exceptions
    if last_index not in ['probe', 'component', None]:
        raise ValueError(f'Argument "last_index" must be one of "probe", "component", or None; "{last_index}" provided.')
        
    if not os.path.exists(path_and_name):
        raise Exception(f'File path specified by user does not exist. ({path_and_name})')
    
    # Process probe data
    file = open(path_and_name, 'r')
    lines = file.readlines()
    file.close()

    time, data, timestep = [], [], []

    for i, line in enumerate(lines):
        line_split = line.split()
        
        try:
            dtype = np.float32 if precision == 'single' else np.float64
            values = list(map(lambda x: dtype(x), line_split))
        except:
            print(f'Entry in line {i} cannot be cast to {dtype}: "{line_split}"')

        if len(timestep) == 0:
            time.append(values[0])
            timestep = timestep + values[1:]
        else:
            timestep = timestep + values

        if len(values) != 9:
            if last_index == 'probe':
                x = timestep[::3]
                y = timestep[1::3]
                z = timestep[2::3]
                data.append([x, y, z])
            
            elif last_index == 'component':
                data.append(np.reshape(timestep, (-1, 3)))
                
            elif last_index == None:
                data.append(timestep)
            
            timestep = []

    return np.array(time), np.array(data)


def convert_distributed_probe(path_and_name, fname=None, precision='single'):
    """Flattens distributed probe file to have one time step per line.
    
    This makes the data more readable for NumPy and similar tools by
    shaping the file into easily identifiable, evenly shaped time steps.

    Parameters
    ----------
    path_and_name : str
        Path to probe file (with .dat suffix)
    fname : str (optional)
        Name of new formatted probe file (saved to same directory)
    precision : str (optional)
        Precision of simulation results ('single' | 'double')

    Returns
    -------
    None
    """
    
    # Obtain flattened array
    t, data = load_distributed_probe(path_and_name, last_index=None, precision=precision)
    array = np.concatenate([t[:,np.newaxis], data], axis=1)
    
    # Save to file    
    if fname == None:
        # TODO: make hardcoded slice more flexible?
        save_path_and_name = path_and_name[:-4] + '_' + str(time.time()) + '.dat'
    else:
        save_path_and_name = '\\'.join(path_and_name.split('\\')[-1] + [fname])
        
    fmt = '%.7E' if precision == 'single' else '%.15E'
    np.savetxt(save_path_and_name, array, fmt=fmt)