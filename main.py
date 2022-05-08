if __name__ == '__main__':
    import argparse
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as utils
    from tqdm import tqdm
    
    # define functions

    def squash(param, p_min, p_max):
        squashed_param_tensor =torch.clamp(param, min=p_min, max=p_max)
        unsqueezed_param = squashed_param_tensor.unsqueeze(1)
        return unsqueezed_param

    def apply_mask(array, flattened_mask, data_shape):
        flattened_list = []
        n=0
        for i in range(len(flattened_mask)):
            if flattened_mask[i][0]==0.0:
                flattened_list.append(float(0))
            else:
                flattened_list.append(array[n][0])
                n+=1
        flattened_array = np.array(flattened_list)
        reshaped_array = np.reshape(flattened_array, data_shape)
        return reshaped_array

    def apply_mu_mask(mu_array, flattened_mask, data_shape):
        flattened_theta= []
        flattened_phi = []
        n=0
        for i in range(len(flattened_mask)):
            if flattened_mask[i][0]==0.0:
                flattened_theta.append(float(0))
                flattened_phi.append(float(0))
            else:
                flattened_theta.append(mu_array[n][0])
                flattened_phi.append(mu_array[n][1])
                n+=1
        flattened_theta_array = np.array(flattened_theta)
        flattened_phi_array = np.array(flattened_phi)
        reshaped_theta = np.reshape(flattened_theta_array, data_shape)
        reshaped_phi = np.reshape(flattened_phi_array, data_shape)
        return reshaped_theta, reshaped_phi

    def ball(ti, t1_, bvals, lambda_iso):
        return abs(1 - (2*np.exp(-ti/t1_)) + np.exp(-7.5/t1_))*np.exp(-bvals * lambda_iso)

    def stick(ti, t1_, bvals, lambda_par, n, mu):
        return abs(1 - (2*np.exp(-ti/t1_)) + np.exp(-7.5/t1_))*np.exp(-bvals * lambda_par * np.dot(n, mu).transpose() ** 2)
        
    def ballstick(ti, t1_ball, t1_stick, bvals, lambda_iso, lambda_par, n, mu, Fp, s0):
        return s0*(Fp*(ball(ti, t1_ball, bvals, lambda_iso)) + (1-Fp)*(stick(ti, t1_stick, bvals, lambda_par, n, mu)))

    def cart2mu(xyz):
        shape = xyz.shape[:-1]
        mu = np.zeros(np.r_[shape, 2])
        r = np.linalg.norm(xyz, axis=-1)
        mu[..., 0] = np.arccos(xyz[..., 2] / r)  # theta
        mu[..., 1] = np.arctan2(xyz[..., 1], xyz[..., 0])
        mu[r == 0] = 0, 0
        return mu

    # Retrieve command line arguments

    parser = argparse.ArgumentParser(description= 'Ball and stick model')
    parser.add_argument('--trainset', '-trs', type=int, help='Input training set, eg 1 or 2')
    parser.add_argument('--prenormalised', '-prn', type=str, help='Type yes to use data normalised w.r.t. highest TI b=0 image')
    parser.add_argument('--learningrate', '-lr', type=float, help='Learning rate')
    parser.add_argument('--batchsize', '-bs', type=int, help='Batch size')
    parser.add_argument('--patience', '-p', type=int, help='Patience')
    parser.add_argument('--dropout', '-d', type=float, help='Dropout (0-1)')
    parser.add_argument('--s0', '-s', type=str, help='yes for s0, no for without')
    args = parser.parse_args()

   
    path = '/Users/jpl/documents/code/ibsc_project/datasets/'

    dataset1 = np.load(path+"dataset1.npz")
    dataset2 = np.load(path+"dataset2.npz")
    dataset3 = np.load(path+"dataset3.npz")
    dataset4 = np.load(path+"dataset4.npz")
    dataset5 = np.load(path+"dataset5.npz")


    dataset_dict = {"1": dataset1, "2": dataset2, "3": dataset3, "4": dataset4, "5": dataset5}
    
    trainset = dataset_dict[str(args.trainset)]
    if args.prenormalised == 'no':
        X_train = trainset['X_train_raw']
    else: X_train = trainset['X_train_normalised']
 
    if args.s0 == 'yes':
        num_params = 8
    else: num_params = 7



    class Net(nn.Module):
        def __init__(self, ti_no0, gradient_directions_no0, b_values_no0):
            super(Net, self).__init__()
            self.ti_no0 = ti_no0
            self.gradient_directions_no0 = gradient_directions_no0
            self.b_values_no0 = b_values_no0
            self.fc_layers = nn.ModuleList()
            for i in range(3): 
                self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
            self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), num_params))
            if args.dropout != 0:
                self.dropout = nn.Dropout(args.dropout)
        
        def forward(self, X):
            if args.dropout != 0:
                X = self.dropout(X)
            params = torch.abs(self.encoder(X))
            t1_ball_uns = params[:, 0]
            t1_ball = squash(t1_ball_uns, 0.010, 5.0)
            t1_stick_uns = params[:, 1]
            t1_stick = squash(t1_stick_uns, 0.010, 5.0)
            lambda_par_uns = params[:, 2]
            lambda_par = squash(lambda_par_uns, 0.1, 3.0)
            lambda_iso_uns = params[:, 3]
            lambda_iso = squash(lambda_iso_uns, 0.1, 3.0)
            Fp = params[:,6].unsqueeze(1)
            theta = params[:,4].unsqueeze(1)
            phi = params[:,5].unsqueeze(1)
            mu_cart = torch.zeros(3,X.size()[0])
            sintheta = torch.sin(theta)
            mu_cart[0,:] = torch.squeeze(sintheta * torch.cos(phi))
            mu_cart[1,:] = torch.squeeze(sintheta * torch.sin(phi))
            mu_cart[2,:] = torch.squeeze(torch.cos(theta))
            if args.s0 == 'yes':
                s0 = params[:,7].unsqueeze(1)
            else: s0 = torch.ones_like(t1_ball)
            mm_prod =  torch.einsum("ij,jk->ki",self.gradient_directions_no0, mu_cart)
            X = (Fp*(torch.abs(1 - (2*torch.exp(-self.ti_no0/t1_ball)) + torch.exp(-7.5/t1_ball))*torch.exp(-self.b_values_no0 * lambda_iso)) + (1-Fp)*(torch.abs(1 - (2*torch.exp(-self.ti_no0/t1_stick)) + torch.exp(-7.5/t1_stick))*torch.exp(-self.b_values_no0 * lambda_par * mm_prod ** 2)))*s0
            return X, t1_ball, t1_stick, lambda_par, lambda_iso, mu_cart, Fp, s0

    def train_model():
        b_values = trainset['b_values']
        ti = trainset['ti']
        n = trainset['gradient_directions']
        b_values_no0 = torch.FloatTensor(b_values)
        ti_no0 = torch.FloatTensor(ti)
        gradient_directions_no0 = torch.FloatTensor(n)
        net = Net(ti_no0, gradient_directions_no0, b_values_no0)
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr = args.learningrate)  
        batch_size = args.batchsize
        num_batches = len(X_train) // batch_size
        trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                                        batch_size = batch_size, 
                                        shuffle = True,
                                        num_workers = 2,
                                        drop_last = True)
        best = 1e16
        num_bad_epochs = 0
        patience = args.patience

        for epoch in range(1000): 
            print("-----------------------------------------------------------------")
            print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
            net.train()
            running_loss = 0.

            for i, X_batch in enumerate(tqdm(trainloader), 0):
                optimizer.zero_grad()
                X_pred, t1_ball_pred, t1_stick_pred, lambda_par_pred, lambda_iso_pred, mu_pred, Fp_pred, s0_pred = net(X_batch)
                loss = criterion(X_pred.type(torch.FloatTensor), X_batch.type(torch.FloatTensor))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            
            print("Loss: {}".format(running_loss))

            if running_loss < best:
                print("############### Saving good model ###############################")
                final_model = net.state_dict()
                best = running_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == patience:
                    print("Done, best loss: {}".format(best))
                    break
                print("Done")
        net.load_state_dict(final_model)

    train_model()
