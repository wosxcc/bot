


blocks =[
    {'name':'fnet1','net':[(64, 16, 1),(64, 16, 1),(64, 16, 2)]},
    {'name':'fnet2','net':[(128, 32, 1),(128, 32, 1),(128, 32, 1),(128, 32, 2)]},
    {'name':'fnet3','net':[(256, 64, 1),(256, 64, 1),(256, 64, 1),(256, 64, 1),(256, 64, 1),(256, 64, 2)]},
    {'name':'fnet4','net':[(512, 256, 1),(512, 256, 1),(512, 256, 2)]}
]

for block in blocks:
    print(block['name'])
    for unit in block['net']:
        uit_depth, uit_depth_bottleneck, unit_stride = unit
        print(uit_depth, uit_depth_bottleneck, unit_stride)