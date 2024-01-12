import numpy as np

SOURCE_NAMES = ['Orientation: +Z; E: +X',
                'Orientation: +Z; E: +Y',
                'Orientation: -X; E: +Z',
                'Orientation: -X; E: +Y',
                'Orientation: -Z; E: +Y',
                'Orientation: -Z; E: +X',
                'Orientation: +X; E: +Z',
                'Orientation: +X; E: +Y',
                'Orientation: -Y; E: +Z',
                'Orientation: -Y; E: +X',
                'Orientation: +Y; E: +X',
                'Orientation: +Y; E: +Z',
                ]

SOURCE_VALUES  = ['0.0000000000E+000    0.0000000000E+000    1.5707963268E+000    0.0000000000E+000',
                  '0.0000000000E+000    0.0000000000E+000    1.5707963268E+000    1.5707963268E+000',
                  '1.5707963268E+000    3.1415926536E+000    0.0000000000E+000    1.5707963268E+000',
                  '1.5707963268E+000    3.1415926536E+000    1.5707963268E+000    1.5707963268E+000',
                  '3.1415926536E+000    0.0000000000E+000    1.5707963268E+000    1.5707963268E+000',
                  '3.1415926536E+000    0.0000000000E+000    1.5707963268E+000    0.0000000000E+000',
                  '1.5707963268E+000    0.0000000000E+000    0.0000000000E+000    3.1415926536E+000',
                  '1.5707963268E+000    0.0000000000E+000    1.5707963268E+000    1.5707963268E+000',
                  '1.5707963268E+000    4.7123889804E+000    0.0000000000E+000    3.1415926536E+000',
                  '1.5707963268E+000    4.7123889804E+000    1.5707963268E+000    0.0000000000E+000',
                  '1.5707963268E+000    1.5707963268E+000    1.5707963268E+000    0.0000000000E+000',
                  '1.5707963268E+000    1.5707963268E+000    0.0000000000E+000    0.0000000000E+000'
                  ]


@staticmethod
def set_planewave(path, source_value):
    ### Modifies source definitions 

    for i, folder in enumerate(FOLDERS):
        emin_file = open('\\'.join([BASEPATH, folder, NAME]), 'r')
        lines = emin_file.readlines()
        emin_file.close()

        for j, line in enumerate(lines):
            if line == '!PLANE WAVE SOURCE\n':
                print('Modifying file ', i+1)

                lines[j + 5] = SOURCE_VALUES[i] + '\n'

                if '* Orientation' in lines[j + 6]:
                    lines[j + 6] = '* ' + SOURCE_NAMES[i] + '\n'
                else:
                    lines.insert(j + 6, '* ' + SOURCE_NAMES[i] + '\n')

                break

        emin_file = open('\\'.join([BASEPATH, folder, NAME]), 'w')
        emin_file.writelines(lines)
        emin_file.close()


@staticmethod
def modify_currents():

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