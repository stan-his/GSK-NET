import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler



# Network class
class GSNet(nn.Module):
    
    def __init__(self, input_dim, output_dim,
                 layer_sizes, kernel_sizes, temp_file='.best-model-parameters.pt'):
        
        
        super(GSNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temp_file = temp_file
        
        self.layers = []
        self.dos = []
        
        d = input_dim[-1]
        
        # Add layers and dropout
        for ls, ks in zip(layer_sizes, kernel_sizes):
            
            self.layers.append(nn.Conv2d(d, ls, ks))
            self.dos.append(nn.Dropout2d(0.3))
            d  = ls 
        
        self.layers = nn.ModuleList(self.layers)
        self.dos = nn.ModuleList(self.dos)

        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001) # 0.0001 works
    
    
        self.start_out = int((self.input_dim[0] - np.sum([x-1 for x in kernel_sizes]) - self.output_dim[0]) / 2)
    
    
    def forward(self, X):
        
        out = X
        
        for l,do in zip(self.layers[:-1], self.dos):
            out = nn.LeakyReLU(0.1)(l(out))
            if self.training:
                out = do(out)
            
        out = nn.Sigmoid()(self.layers[-1](out))

        i1,i2 = self.start_out, self.start_out + self.output_dim[0]
        j1,j2 = self.start_out, self.start_out + self.output_dim[1]
        
        return out[:,:,i1:i2,j1:j2]



    
    def train_net(self, data, batch_size=256, epochs=500, splits=None):
    
        self.train(True)
    
        self.best_val = 9999999
        self.saved_model = None
        
        losses = [[],[]]
        
    
        indx = np.arange(len(data))
    
    
        if splits is None:
            r = np.random.rand(len(data))
            indx_train = indx[r < 0.8]
            indx_val = indx[r >= 0.8]
            
        else:
            indx_train = indx[splits==0]
            indx_val = indx[splits==1]
            
        
        
        train_sampler = SubsetRandomSampler(indx_train)
        val_sampler = SubsetRandomSampler(indx_val)
        
        train_generator = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        val_generator = torch.utils.data.DataLoader(data, batch_size=32, sampler=val_sampler)
        
        

        for e in range(epochs):
            if e % 1 == 0:
                print(e)
            
            ls = []
            vls = []
            
            for b_x, b_y in train_generator:
                
                self.optimizer.zero_grad()

                y_hat = self.forward(b_x)
                
                loss = self.loss(y_hat, b_y)
                loss.backward()
                self.optimizer.step()
    
                ls.append(loss.cpu().detach())
        
            
            with torch.no_grad():
                for b_x, b_y in val_generator:
                    y_hat = self.forward(b_x)
                    loss = self.loss(y_hat, b_y)
                    vls.append(loss.cpu().detach())
                    

            
            tl = torch.mean(torch.tensor(ls))
            vl = torch.mean(torch.tensor(vls))
            
            print(tl, vl)
            
            losses[0].append(tl)
            losses[1].append(vl)
            
            if vl < self.best_val:
                self.best_val = vl
                self.saved_model = self.state_dict().copy()
                torch.save(self.state_dict(), self.temp_file)
            
            
        self.load_state_dict(self.saved_model)    
            
        return losses
    
    # Evaluate the performance of the network
    def eval_net(self, data):
        
        self.eval()
        run = True
        iters = 1
        
        while run:
            iters += 1
            
            generator = torch.utils.data.DataLoader(data, batch_size=128)

            with torch.no_grad():
                    for i,b_x in generator:
                        y_hat = self.forward(b_x)
                        data.write_indx(i.numpy(), y_hat.cpu().numpy())

            
            del generator
            torch.cuda.empty_cache()
            run = data.load_next()
            time.sleep(0.5)
        
        
        
        data.final_eval()
    
    
    def load_model(self, file_name=None):
        
        if file_name is None:
            file_name = self.temp_file
        
        self.load_state_dict(torch.load(file_name))
        self.eval()

        
        
        
