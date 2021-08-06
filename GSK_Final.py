import os
import torch
import rasterio
import matplotlib.pyplot as plt
import numpy as np


from Parameters import pixels, pad, device, layers, strides, input_tif, output_tif
from TrainDataset import TrainDataset
from GSNet import GSNet
from datetime import datetime


im_size = [2*pad + pixels, 2*pad + pixels, 3]


    
# Setup dataset for training
src = rasterio.open(input_tif)

X = np.asarray([src.read(i+1) for i in range(3)])
X = np.moveaxis(X, 0, -1)

src = rasterio.open(output_tif)
Y = src.read(1)

data = TrainDataset(X,Y, pixels, pad)

    
# Split data into folds

xm,ym = [x/3 for x in X.shape[:2]]

split = np.asarray([
    x // xm + (y // ym) * 3 + \
    (5 if (x // xm + (y // ym) * 3) == 4 and (2*x > X.shape[0]) else 0)
    for x,y in data.indx
])




# Move data to GPU
data.to_GPU(device)


# Log start
now = datetime.now()
now = now.strftime("%d/%m/%Y %H:%M:%S")

with open("Training_log.txt", "w+") as f:
    f.write(f"Start run at: {now}\n")



    
# For each fold
for i in range(10):

    net = GSNet(im_size, (pixels, pixels, 1), layers, strides)
    net.to(device)

    sp = np.zeros_like(split)
    sp[split == (i + 1) % 9] = 1
    sp[split == i] = 2

    losses = net.train_net(data, batch_size=128, epochs=150, splits=sp)

    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")

    with open("Result2_log.txt", "a+") as f:
        f.write(f"\n{'#' * 80}\n")
        f.write(f"\nFold {i} done at: {now}\n")

    
    
    net.to("cpu")
    torch.save(net.state_dict(), f"Models/best-model-parameters-{i}.pt")
    
    
    net.to(device)
    net.eval()

    test_inx = np.arange(len(data))[split==2]

    tp, fp, tn, fn = [torch.tensor(0.0) for i in range(4)]

    net.to(device)
    net.eval()

    for i in test_inx:
        y = (data[i][1][None,:,:,:]).cpu()
        yh = (net(data[i][0][None,:,:,:]).cpu() > 0.5) * 1

        tp += ((y == 1) & (yh == 1)).sum()
        fp += ((y == 0) & (yh == 1)).sum()
        fn += ((y == 1) & (yh == 0)).sum()
        tn += ((y == 0) & (yh == 0)).sum()

    
    rec = (tp / (1. * (tp + fp )))
    prec = (tp / (1. * (tp + fn )))

    with open("Training_log.txt", "a+") as f:
        f.write(f"Precision = {prec}\n")
        f.write(f"Recall = {rec}\n")
        f.write(f"F1 = {2 * (prec * rec) / (prec + rec)}\n")

    
    plt.plot(np.asarray(losses).T)
    plt.savefig(f"Figs/losses{i}.png")
    
    
    
    # Clean up memory
    del net
    torch.cuda.empty_cache()
    
    
# Log end
now = datetime.now()
now = now.strftime("%d/%m/%Y %H:%M:%S")

with open("Training_log.txt", "a+") as f:
    f.write(f"\n\nRun ended at: {now}")
