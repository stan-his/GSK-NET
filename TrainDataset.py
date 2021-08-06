import torch
import numpy as np

class TrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, Y, pixels, pad):
        
        self.pad = pad
        self.pixels = pixels        
        self.X = X / 255.
        self.Y = Y
        self.indx = []
        self.gpu = False

        for xc in np.arange(self.pad, X.shape[0]-self.pixels-self.pad, self.pixels * 2/3):
            for yc in np.arange(self.pad, X.shape[1]-self.pixels-self.pad, self.pixels * 2/3):        
                xc, yc = int(xc), int(yc)
                x = X[xc-self.pad:xc+self.pixels+self.pad, yc-self.pad:yc+self.pixels+self.pad]
                if ((x > 0) & (x < 255)).mean() > 0.4:
                    self.indx.append((xc,yc))
        
        
        
        
    def to_GPU(self, device="cuda"):
        
        self.gpu = True
        self.X = torch.tensor(self.X, device=device, dtype=torch.float32).permute([2,0,1])
        self.Y = torch.tensor(self.Y, device=device, dtype=torch.float32)[:,:,None].permute([2,0,1])

        
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indx)

    def __getitem__(self, index):
      
        xc, yc = self.indx[index]
        
        if not self.gpu:
            X = self.X[xc-self.pad:xc+self.pixels+self.pad, yc-self.pad:yc+self.pixels+self.pad]
            Y = self.Y[xc:xc+self.pixels, yc:yc+self.pixels]
        else:
            X = self.X[:,xc-self.pad:xc+self.pixels+self.pad, yc-self.pad:yc+self.pixels+self.pad]
            Y = self.Y[:,xc:xc+self.pixels, yc:yc+self.pixels]
            
        return X, Y