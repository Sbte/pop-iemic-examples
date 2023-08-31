top_layer = []
nx = 240
ny = 108
nz = 12

start = nx // 15
west_na = int(nx / 1.375)
east_na = int(nx / 1.2)

with open(f'global_{nx}x{ny}x{nz}.mask') as f:
    # First parse the top layer
    started = False
    for l in f.readlines():
        if l.startswith(f'% {nx} {ny} {nz} 13'):
            started = True
            continue

        if not started:
            continue

        if l.startswith('%'):
            break

        end = west_na
        for i in range(west_na, east_na):
            if l[i] == '1':
                break
            end += 1
        top_layer.append(l[:start] + '1' * (end - start) + l[end:])

new = ''
with open(f'global_{nx}x{ny}x{nz}.mask') as f:
    # Now apply the top layer mask to all other layers
    for l in f.readlines():
        if l.startswith('%'):
            i = 0
        else:
            for j, c in enumerate(l):
                if top_layer[i][j] == '1':
                    new += '1'
                else:
                    new += c
            i += 1
            continue

        new += l

with open(f'amoc_{nx}x{ny}x{nz}.mask', 'w') as f:
    f.write(new)
