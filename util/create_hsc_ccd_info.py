"""
Program for reading HSC detector info from a text file for the purpose
of generating a module that can be imported into the HSCPlanner Ginga
plugin for drawing the detector pattern outlines.
"""
import sys
import re
import pprint


def read_ccd_polygon_coords(filepath):

    with open(filepath, 'r') as in_f:
        lines = in_f.read().split('\n')

    pattern = r'^ccd:\s+(\d+)\s+\(dRa,dDec\):\s+([\d\.-]+)\s+([\d\.-]+)\s*$'
    info = {}
    for line in lines:
        match = re.match(pattern, line)
        if match:
            ccd = int(match.group(1))
            dra = float(match.group(2))
            ddec = float(match.group(3))
            d = info.setdefault(ccd, {})
            l = d.setdefault('polygon', [])
            l.append((dra, ddec))

    return info

def write_python_ccd_polygon_coords(filepath, info):

    keys = list(info.keys())
    keys.sort()

    with open(filepath, 'w') as out_f:
        out_f.write("info = ")
        pprint.pprint(info, stream=out_f)



def main(options, args):
    infile = args[0]
    outfile = args[1]

    #info = read_ccd_polygon_coords(infile)
    import hsc_ccd_info
    info = hsc_ccd_info.info

    from naoj.hsc import hsc_dr

    #colors = ['green', 'skyblue', 'orange', 'purple', 'tan', 'cyan', 'gold']
    keys = list(info.keys())
    keys.sort()
    for i in range(len(keys)):
        ## info[keys[i]]['color'] = colors[i % len(colors)]
        det_id = keys[i]
        addl = hsc_dr.ccd_aux_info1[det_id]
        if 'bad_channels' in addl:
            info[det_id]['color'] = 'red'
            info[det_id]['bad_channels'] = addl['bad_channels']
        elif addl['bee_id'] == 0:
            info[det_id]['color'] = 'skyblue'
        elif addl['bee_id'] == 1:
            info[det_id]['color'] = 'violet'

    write_python_ccd_polygon_coords(outfile, info)

if __name__ == '__main__':
    main(None, sys.argv[1:])
