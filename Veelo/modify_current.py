import sys
import os
import glob
import numpy as np


if len(sys.argv) > 1:
    print(f'\tSearching directory "{sys.argv[-1]}"')
    os.chdir(sys.argv[-1])
else:
    print('\tSearching current directory')
   
if len(glob.glob('*.emin')) > 0:
    emin_name = glob.glob('*.emin')[0]
else:
    print('\tNo emin file found in specified directory.')
    
emin = open(emin_name, 'r')
lines = emin.readlines()
emin.close()

i0, i1 = 0, 0

for i, line in enumerate(lines):
    if 'SourceCurrent.dat' in line:
        i0 = i + 1
        
    elif 'PROBES' in line:
        i1 = i - 2
        break
        
        
probe = [line for line in lines[i0:i1] if float(line.split()[4]) != 0]

new_lines = lines[:i0] + probe + lines[i1:]

emin = open(emin_name, 'w')
emin.writelines(new_lines)
emin.close()

print('\tSaved modified emin file.')