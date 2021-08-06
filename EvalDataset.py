import rasterio
import numpy as np
import torch

class EvalDataset(torch.utils.data.Dataset):
    
    def __init__(self, fig_name, pixels, pad, gray=True):
        
        super(EvalDataset, self).__init__()
        
        self.pad = pad
        self.pixels = pixels
        self.keep = 0.2 
        self.indx = []        
        self.stack = []
        self.lims = []
        self.x_off = 0
        self.y_off = 0
        self.device = "cpu"
        self.org_data = None
        
        
        
        src = rasterio.open(fig_name)
        input_data = np.asarray([src.read(x+1) for x in range(src.count)])
        input_data = np.moveaxis(input_data, 0, -1)
        
        self.transform = src.transform
        self.crs = src.crs

        
        self.data = input_data.mean(-1, keepdims=True) if gray else input_data
        self.out_data = np.zeros(self.data.shape[:2], dtype="float32")
        self.writes = np.zeros_like(self.out_data)
        self.eval_time = False
        self.shape = self.data.shape
        
        
        
        self.data = self.data / 255.
        
        
        # Will not fit in GPU memory
        if np.prod(self.data.shape[:2]) > 4e8:
            
            self.org_data = self.data.copy()
            
            split = 8500
            
            for i in range(0, self.data.shape[0], split):
                for j in range(0, self.data.shape[0], split):
                    self.stack.append(self.data[max(i-pad, 0):i+split+pad,
                                           max(j-pad, 0):j+split+pad])
                    self.lims.append((max(i-pad, 0), max(j-pad, 0)))
        
        
            self.load_next()
        
        # Fit in GPU memory
        else:
            self.setup_index(0,0)
                                   
                                 
    
    def setup_index(self, x_off, y_off):
                                      
        self.x_off = x_off
        self.y_off = y_off
        self.indx = []
        
        for xc in np.arange(self.pad, self.data.shape[0]-self.pixels-self.pad, self.pixels):            
            for yc in np.arange(self.pad, self.data.shape[1]-self.pixels-self.pad, self.pixels):
                xc, yc = int(xc), int(yc)
                
                x = self.data[xc-self.pad:xc+self.pixels+self.pad, yc-self.pad:yc+self.pixels+self.pad]
                if ((x > 0) & (x < 255)).mean() > 0.2:
                    self.indx.append((xc,yc))

                                 
                                 
        
    def prepare_for_net(self, device="cpu"):
        
        self.eval_time = True
        self.device = device
        self.data = torch.tensor(self.data, device=device, dtype=torch.float32).permute([2,0,1])
        
    
    
    def back_from_net(self):
        self.eval_time = False
        self.data = self.data.permute([1,2,0]).cpu().numpy()

    
    def load_next(self):
        
        if len(self.stack) == 0:
            return False
        
        del self.data
        torch.cuda.empty_cache()
        
        self.data = self.stack.pop()
        ls = self.lims.pop()
        self.setup_index(*ls)
        
        
        if self.eval_time:
            self.prepare_for_net(self.device)
        
        if len(self.indx) == 0:
            self.load_next()
        
        return True
    
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indx)

    
    def __getitem__(self, index):
      
        xc, yc = self.indx[index]
        
        if self.eval_time:
            return index, self.data[:,xc-self.pad:xc+self.pixels+self.pad, yc-self.pad:yc+self.pixels+self.pad]
        else:
            return self.data[xc-self.pad:xc+self.pixels+self.pad, yc-self.pad:yc+self.pixels+self.pad]

    
    def write_indx(self, index, data):
        
        for n,i in enumerate(index):
            xc, yc = self.indx[i]
            xc += self.x_off
            yc += self.y_off
#             x = np.moveaxis(data[n], 0, -1)
            x = data[n][0]
            self.out_data[xc:xc+x.shape[0], yc:yc+x.shape[1]] += x
            self.writes[xc:xc+x.shape[0], yc:yc+x.shape[1]] += 1
            
    
    
    def final_eval(self):
        if not self.org_data is None:
            self.data = self.org_data
            if self.eval_time:
                self.prepare_for_net(self.device)
            
        self.out_data[self.writes > 0] = self.out_data[self.writes > 0] / self.writes[self.writes > 0]
        self.writes = 1 * (self.writes > 0)
        
        