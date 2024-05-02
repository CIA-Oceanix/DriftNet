import torch.nn as nn
import torch


#%%
class DriftNetDATARMOR(nn.Module):
        def __init__(self):
            super(DriftNetDATARMOR, self).__init__()

            self.first_cond_input = (N_var_cond + 1) * 9
            self.days = days
            self.first_conv_kernel_spatial = 3 
            self.first_conv_kernel_temporal = 5
            self.hidden_state = 32 * N_var_cond
            self.first_cond_output = self.hidden_state
            self.batch_size = batch_size

            self.conv_layer1 = self._conv_layer_set(self.first_cond_input, self.first_cond_output, self.first_conv_kernel_temporal, self.first_conv_kernel_spatial, self.first_conv_kernel_spatial)  # avant -> 4, 4

            self.CONV_LSTM_layer = ConvLSTM(9, 48, (7, 7), 1, batch_first=True, bias=True, return_all_layers=False)

            self.cnn = nn.Sequential(
                nn.Conv2d(self.hidden_state * 48, 48, kernel_size = (2, 2), stride = 1, padding = 'same', padding_mode = 'reflect', bias = False), # 2, 2 before
            )

        def _conv_layer_set(self, in_c, out_c, k1, k2,k3):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_c, out_c * 9, kernel_size=(k2, k3), padding = 'same', padding_mode = 'reflect', bias = False),
                nn.LeakyReLU(True),
            )
            return conv_layer

        def pdf(self,out):
            out = self.cnn(out)
            return out   

        def temporal_lstm_loop(self, out, previous):
            out_temporal, hidden = self.CONV_LSTM_layer(out, previous)
            return out_temporal, hidden


        def forward(self, cond_input):
            out = self.conv_layer1(cond_input.view(cond_input.size()[0], -1, *cond_input.size()[-2:]))
            out_temporal, hidden = self.temporal_lstm_loop(out.view(cond_input.size()[0], self.hidden_state, self.days, *cond_input.size()[-2:]), None)
            out = self.pdf(out_temporal[0].view(cond_input.size()[0], -1, *cond_input.size()[-2:]))
            return out.unsqueeze(1) #out[:,:,1:], out_CNN #.a    transpose(1, 2) 

        
    def gridPDF_to_gridLatLon(out, N_uv, coordinates):
        out_final = torch.zeros((batch_size, 2, 36)).cuda()
        dx = 1/12
        dy = 1/12
        for time_step in range(9 * 4):
            x = [torch.arange(torch.min(x[0], -1)[0], torch.min(x[0], -1)[0] + N_uv * dx, dx).cuda()[:N_uv] for x in coordinates]
            y = [torch.arange(torch.min(x[1], -1)[0], torch.min(x[1], -1)[0] + N_uv * dy, dy).cuda()[:N_uv] for x in coordinates]
            grid_XXgridYY = [torch.meshgrid(xx, yy, indexing='ij') for xx, yy in zip(x, y)]
            x = torch.cat([torch.sum(torch.mul(out_individual[:,time_step], x[0]), (1,2)) for x, out_individual in zip(grid_XXgridYY, out)]).unsqueeze(1)
            y = torch.cat([torch.sum(torch.mul(out_individual[:,time_step], x[1]), (1,2)) for x, out_individual in zip(grid_XXgridYY, out)]).unsqueeze(1)
            out_final[:,:,time_step] = torch.cat([x,y], 1) #for xx, yy in zip(x, y)])
    return out_final[:,:,1:]

