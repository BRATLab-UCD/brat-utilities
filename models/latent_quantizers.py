import torch

class SoftQuantize(torch.nn.Module):
    def __init__(self, r, L, m, sigma=1.0, hard_sigma=1e6, sigma_trainable=True, bs=200, device="cpu"):
        """
        Soft quantization layer based on Voronoi tesselation centers
        sigma = temperature for softmax
        r = dimensionality of autoencoder's latent features
        L = number of cluster centers
        m = dimensionality of cluster centers
        """
        super(SoftQuantize, self).__init__()
        self.sigma_trainable = sigma_trainable
        # self.sigma = torch.nn.Parameter(data=torch.Tensor([sigma]).to(device), requires_grad=True) if sigma_trainable else sigma.to(device)
        self.sigma = sigma
        self.hard_sigma = 1e6
        self.r = r
        self.L = L # num centers
        self.m = m # dim of centers
        self.n_features = int(r/m)
        self.softmax = torch.nn.Softmax(dim=2)
        self.sigma_eps = 1e-4
        self.quant_mode = 0 # 0 = pass through (no quantization), 1 = soft quantization, 2 = return softmax outputs, 3 = hard quantization with self.hard_sigma
        self.q_hot_template = torch.zeros(bs, self.n_features, self.L).to(device)
        
    def init_centers(self, c):
        self.c = torch.nn.Parameter(data=c, requires_grad=True)
        self.quant_mode = 1

    def forward(self, x):
        """
        Take in latent features z, return soft assignment to clusters
        """
        b = x.shape[0]
        if self.quant_mode == 0:
            return x # no quantization in latent layer
        else:
            z = x.view(b, self.n_features, self.m, 1).repeat(1, 1, 1, self.L)
            c = self.c.view(1,1,self.m,self.L).repeat(b, self.n_features, 1, 1)
            if self.sigma_trainable:
                sigma = self.sigma.relu() + self.sigma_eps 
            elif self.quant_mode == 3:
                sigma = self.hard_sigma
            else:
                sigma = self.sigma
            q = self.softmax(-sigma*torch.sum(torch.pow(z - c, 2), 2))
            if self.quant_mode == 2:
                return q # softmax outputs
            c = self.c.view(1,self.m,self.L).repeat(b,1,1).transpose(2,1)
            out = torch.matmul(q, c)
            return out.view(b, self.r) # soft quantized outputs

class SoftQuantizeMCR(SoftQuantize):
    def __init__(self, *args, **kwargs):
        """
        extend forward call fo  SoftQuantize class to work with MCR2 loss
        from Yu, Yaodong, Chan, Kwan Ho Ryan, You, Chong, Song, Chaobing, and Ma, Yi
        "Learning diverse and discriminative representations via the principle of 
        maximal coding rate reduction." 2020, Advances in Neural Information Processing
        Systems, Vol. 33.
        """
        SoftQuantize.__init__(self, *args, **kwargs)
        self.mcr_bool = True
    
    def forward(self, x):
        """
        Take in latent features z, return soft assignment to clusters and reshaped softmax output

        $\mathbf \Pi$ is the partition matrix
        """
        b = x.shape[0]
        z = x.view(b, self.n_features, self.m, 1).repeat(1, 1, 1, self.L)
        c = self.c.view(1,1,self.m,self.L).repeat(b, self.n_features, 1, 1)
        if self.quant_mode == 3:
            sigma = self.hard_sigma
        else:
            sigma = self.sigma
        q = self.softmax(-sigma*torch.sum(torch.pow(z - c, 2), 2))
        c = self.c.view(1,self.m,self.L).repeat(b,1,1).transpose(2,1)
        out = torch.matmul(q, c)
        # Pi = torch.diag_embed(q.transpose(1,2)) # reshape to (b, L, m, m) [?]
        # print(f"-> Pi.size()={Pi.size()} -> sum batches (m={Pi.size(-1)}): {torch.sum(Pi.view(b,-1), axis=1)}")
        return [out.view(b, self.r), x.view(b, self.n_features, self.m), q] # soft quantized outputs and softmax output

def energy_loss(y_gt, y_hat):
    return torch.mean(torch.sum(torch.pow(y_gt - y_hat, 2), dim=1))