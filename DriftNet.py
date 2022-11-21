import torch.nn as nn
import torch

class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3, stochastic=False):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)
        self.stochastic = stochastic
        #self.correlate_noise = CorrelateNoise(input_size, 10)
        #self.regularize_variance = RegularizeVariance(input_size, 10)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]
        #print(spatial_size)
        """
        if self.stochastic == True:
            z = torch.randn(input_.shape).to(device)
            z = self.correlate_noise(z)
            z = (z-torch.mean(z))/torch.std(z)
            #z = torch.mul(self.regularize_variance(z),self.correlate_noise(z))
        """
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)),#.to(device),
                torch.autograd.Variable(torch.zeros(state_size))#.to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        """
        if self.stochastic == False:
            stacked_inputs = torch.cat((input_, prev_hidden), 1)
        else:
            stacked_inputs = torch.cat((torch.add(input_,z), prev_hidden), 1)
        """

        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell
    




class DriftNet(nn.Module):
    def __init__(self):
        super(DriftNet, self).__init__()
        
        self.first_cond_output = 9 * 4 * 11 #64 #11 #more ! WAS 12
        self.first_cond_input_time_dim = 9
        self.first_cond_input = (N_var_cond + 1) * self.first_cond_input_time_dim
        
        self.N_uv = 14
        self.first_conv_kernel_spatial = 5 # 8 avant   ¸ THE LAST MODIF IS HERE 2 -> 1 (1 found previously as the best)
        self.conv_layer1 = self._conv_layer_set(self.first_cond_input, self.first_cond_output, 2, self.first_conv_kernel_spatial, self.first_conv_kernel_spatial)  # avant -> 4, 4
        self.CONV_LSTM_layer = ConvLSTM2d(self.first_cond_output, 36)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(36, 16, kernel_size = (4, 4), stride = 1, padding = 0, bias = False),           
            nn.Softmax(),

        )

        self.last = nn.Sequential(   # first spatial input = 64 OR 36
                                  
            nn.Conv1d(16, 8, kernel_size = 9, stride = 1, padding = 0, bias = False),
            nn.LeakyReLU(True),
            
            nn.Conv1d(8, 4, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.LeakyReLU(True),
            
            nn.Conv1d(4, 2, kernel_size = 4, stride = 1, padding = 0, bias = False),
        )

          
    def _conv_layer_set(self, in_c, out_c, k1, k2,k3):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(k2, k3), padding=(0,0), bias = True),
        nn.LeakyReLU(True),
        )
        return conv_layer
 
    
    def lstm_cond(self, out):
        batch_size = out.shape[0]
        h_0 = torch.autograd.Variable(torch.randn(self.num_layer_COND, batch_size, self.cond_out)).to(dev) # hidden state
        c_0 = torch.autograd.Variable(torch.randn(self.num_layer_COND, batch_size, self.cond_out)).to(dev) # hidden state
        out, h_n = self.lstm_COND(out, (h_0, c_0))

        return out
    
    def FIRST_lstm_cond(self, out):
        batch_size = out.shape[0]
        h_0 = torch.autograd.Variable(torch.randn(self.num_layer_COND, batch_size, self.cond_out)).to(dev) # hidden state
        c_0 = torch.autograd.Variable(torch.randn(self.num_layer_COND, batch_size, self.cond_out)).to(dev) # hidden state
        out, h_n = self.lstm_COND(out, (h_0, c_0))

        return out


    def pdf(self,out):
        out = self.cnn(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.last(out)

        return out   #.transpose(1, 2)   # 64, 32, 64


    def forward(self, cond, init_pos):
      
        out = torch.cat((cond.transpose(1, 2), init_pos.transpose(1, 2)), dim = 1)
        out = out.view(out.shape[0], out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
        out = self.conv_layer1(out)        
        out, _ = self.CONV_LSTM_layer(out, None)
        out = out.reshape(out.shape[0], 36, 10, 10)
        out = self.pdf(out)
        return out, out #.transpose(1, 2) 
