top_layer = []
with open('global_120x54x12.mask') as f:
    # First parse the top layer
    started = False
    for l in f.readlines():
        if l.startswith('% 120 54 12 13'):
            started = True
            continue

        if not started:
            continue

        if l.startswith('%'):
            break

        start = 8
        end = 87
        end2 = 100
        for i in range(end, end2):
            if l[i] == '1':
                break
            end += 1
        top_layer.append(l[:start] + '1' * (end - start) + l[end:])

new = ''
with open('global_120x54x12.mask') as f:
    # Now apply the top layer mask to all other layers
    for l in f.readlines():
        if l.startswith('%'):
            i = 0

        if len(l) > 80:
            for j, c in enumerate(l):
                if top_layer[i][j] == '1':
                    new += '1'
                else:
                    new += c
            i += 1
            continue

        new += l

with open('amoc_120x54x12.mask', 'w') as f:
    f.write(new)
