

# Network parameters

layers = [128, 64, 64, 32, 32, 32, 1]
strides = [9] * 2 + [7] * 3 + [5] * 2


# Image parameters

pixels = 80
pad = sum(strides) / 2 + 1



# Hardware parameters

device = "cuda:2"