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

#%%
class DriftNet(nn.Module):
    def __init__(self, load_checkpoint = False, path = None):
        super(DriftNet, self).__init__()
        
        
        #self.ConvLSTM = conv
        self.first_cond_input = N_var_cond + 1 
        self.N_uv = 14
        self.days = days
        self.first_conv_kernel_spatial = 2 #5 # 8 avant   ¸ THE LAST MODIF IS HERE 2 -> 1 (1 found previously as the best)
        self.hidden_state = 8
        self.first_cond_output = self.hidden_state
        self.first_conv_kernel_temporal = 1
        
        self.interpolate_size = 14
        
        self.size_final = (self.interpolate_size, self.interpolate_size, 35)

        self.conv_layer1 = self._conv_layer_set(self.first_cond_input, self.first_cond_output, self.first_conv_kernel_temporal, self.first_conv_kernel_spatial, self.first_conv_kernel_spatial)  # avant -> 4, 4
        
        self.CONV_LSTM_layer = ConvLSTM2d(self.first_cond_output, self.hidden_state)
        
        if(load_checkpoint == True):
            checkpoint = torch.load(path)
            self.CONV_LSTM_layer.load_state_dict(checkpoint['CONV_LSTM_model_state_dict'])
        
        self.cnn = nn.Sequential(
            nn.Conv3d(self.hidden_state, 1, kernel_size = (1, 1, 1), stride = 1, padding = 0, bias = False),
          
        )
          
    def _conv_layer_set(self, in_c, out_c, k1, k2,k3):
        conv_layer = nn.Sequential(
        
        nn.Conv3d(in_c, out_c, kernel_size=(k1, k2, k3), padding="same", padding_mode = "reflect", bias = True),
        nn.LeakyReLU(True),
        )
        return conv_layer

    def gridPDF_to_gridLatLon(self, out):
        out_final = torch.autograd.Variable(torch.zeros((out.size()[0], 2, 36)))
        dx = 2/(self.size_final[0]-1)
        dy = 2/(self.size_final[1]-1)
        x = torch.arange(-1, 1+dx, dx)
        y = torch.arange(-1, 1+dy, dy)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')   
        for time_step in range(self.days * 4):
            x = torch.sum(torch.mul(out[:,:,time_step], grid_x), (2,3))#.unsqueeze(1)
            y = torch.sum(torch.mul(out[:,:,time_step], grid_y), (2,3))#.unsqueeze(1)
            out_final[:,:,time_step] = torch.cat([x,y], 1)
        return out_final

    def proba_to_pos(self, out):
        out = self.gridPDF_to_gridLatLon(out)
        return out
    
    def pdf(self,out):
        out = self.cnn(out)
        min_value = torch.abs(torch.repeat_interleave(out.view(*out.size()[:3], -1).min(dim = -1)[0].unsqueeze(-1), self.N_uv * self.N_uv).view_as(out))
        #max_value = torch.abs(torch.repeat_interleave(out.view(*out.size()[:3], -1).max(dim = -1)[0].unsqueeze(-1), self.N_uv * self.N_uv).view_as(out))
       
        out = out + min_value #/ max_value
        out = out/torch.repeat_interleave(((out.view(out.size()[0], out.size()[1], out.size()[2], -1)).sum(-1)).unsqueeze(-1), self.N_uv * self.N_uv, dim = -1).view_as(out)
       
        #m = nn.Softmax(3)
        #out = m(out.view(*out.size()[:3], -1).view_as(out))
        
        return out   #.transpose(1, 2)   # 64, 32, 64
    
    
    def ResBlock(self, out, hidden):

        #out_temporal = torch.autograd.Variable(torch.zeros((self.batch_size, self.hidden_state, 1, self.N_uv - self.first_conv_kernel_spatial + 1, self.N_uv - self.first_conv_kernel_spatial + 1)))
        out = self.conv_layer1(out)
        print(hidden)
        out, hidden = self.CONV_LSTM_layer(out.squeeze(), hidden)
        print(out.shape)

        out = self.cnn(out.unsqueeze(2))


        return out, hidden
        
    def CONV_LSTM_layer(self, out):
        out_temporal = torch.autograd.Variable(torch.zeros((out.size()[0], self.hidden_state, self.days * 4, self.N_uv , self.N_uv )))
        #out_temporal_reverse = torch.autograd.Variable(torch.zeros((out.size()[0], self.hidden_state, self.days * 4, self.N_uv , self.N_uv )))
        
        out_CNN = self.conv_layer1(out)
        
        out_interpolated = torch.nn.functional.interpolate(out_CNN, (out_CNN.size()[2]*4, out_CNN.size()[3], out_CNN.size()[-1]))
        
        days_range = np.arange(0, days, 1)
        
        for day in days_range:
            for step in range(4):
                if(day == 0 and step == 0):
                    output_state = self.CONV_LSTM_layer(out_CNN[:,:,day], None)
                else:
                    if(step == 0):
                        output_state = self.CONV_LSTM_layer(out_CNN[:,:,day], output_state)
                    else:
                    
                        output_state = self.CONV_LSTM_layer(out_temporal[:,:,day*4 + step - 1], output_state)
                out_temporal[:,:,day*4+step] = output_state[0].squeeze()
             
        
        for day in range(8, -1):
            for step in range(4, -1):
                if(day == 8 and step == 4):
                    output_state = self.CONV_LSTM_layer(out_CNN[:,:,day], None)
                else:
                    if(day == 8 and step == 3):
                        output_state = self.CONV_LSTM_layer(out_temporal[:,:,day*4 + step + 1], None)
                    else : 
                        output_state = self.CONV_LSTM_layer(out_temporal[:,:,day*4 + step + 1], output_state)
                out_temporal[:,:,day*4+step] += output_state[0].squeeze()
    
        out_CNN = self.pdf(out_temporal)
        
        out = self.proba_to_pos(out_CNN)
        
        return out, out_CNN

    def forward(self, cond, init_pos):
        
        out = torch.cat((cond.transpose(1, 2), init_pos.transpose(1, 2)), dim = 1)
        
        out, out_CNN = self.CONV_LSTM_layer(out)
        
        
        return out[:,:,1:], out_CNN #.transpose(1, 2) 
