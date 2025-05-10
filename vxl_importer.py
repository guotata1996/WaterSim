# goxel https://github.com/guillaumechereau/goxel/releases/tag/v0.15.1

import numpy as np
from collections import defaultdict

def import_vxl(fpath):
    xmin = ymin = zmin = 9999
    xmax = ymax = zmax = -9999
    lines = []
    with open(fpath, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            x, y, z, color = line.split(' ')
            x = int(x)
            y = int(y)
            z = int(z)
            color = color.strip('\n')
            lines.append((x, y, z, color))

    # 000000 determines base size
    for x, y, z, color in lines:
        if color == '000000':
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            zmin = min(zmin, z)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
            zmax = max(zmax, z)

    terrain = np.zeros([xmax - xmin + 1, ymax - ymin + 1])
    # 8f563b is terrain
    for x, y, z, color in lines:
        if color == '8f563b':
            terrain[x - xmin, y - ymin] = max(terrain[x - xmin, y - ymin], z - zmax)

    # 639bff is water
    # fbf236 is source
    water = np.zeros_like(terrain)
    source = defaultdict(lambda : 0)
    for x, y, z, color in lines:
        if color == '639bff' or color == 'fbf236':
            water[x - xmin, y-ymin] = max(water[x - xmin, y-ymin], z - terrain[x - xmin, y - ymin])
        if color == 'fbf236':
            source[(x - xmin, y-ymin)] += 1

    return terrain, water, source
