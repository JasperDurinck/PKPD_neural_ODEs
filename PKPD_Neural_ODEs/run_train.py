
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint
import utils
import numpy as np
import torch


def train_one_epoch(config):

    torch.manual_seed(config.args.seed)
    np.random.seed(config.args.seed)

    epoch = config.args.current_epoch 
    batches_per_epoch = config.data_obj["n_train_batches"]

    print('epoch', epoch)
    for _ in tqdm(range(batches_per_epoch), ascii=True):
        config.optimizer.zero_grad()

        ptnms, times, features, labels, cmax_time = config.data_obj["train_dataloader"].__next__()
        dosing = torch.zeros([features.size(0), features.size(1), config.args.latent_dim]) 
        dosing[:, :, 0] = features[:, :, -1-config.args.output_dim] #AMT must be -2
        dosing = dosing.permute(1, 0, 2)

        times = times.to(device=config.device)

        encoder_out = config.encoder(features) 
        qz0_mean, qz0_var = encoder_out[:, :config.args.latent_dim], encoder_out[:, config.args.latent_dim:] 
        z0 = utils.sample_standard_gaussian(qz0_mean, qz0_var).to(config.device) 

        solves = z0.unsqueeze(0).clone().to(config.device)
        try:
            for idx, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
                z0 += dosing[idx].to(config.device) 
                time_interval = torch.Tensor([time0 - time0, time1 - time0]).to(config.device)
                sol = odeint(config.ode_func, z0, time_interval, rtol=config.args.tol, atol=config.args.tol)
                z0 = sol[-1].clone()
                solves = torch.cat([solves, sol[-1:, :]], 0)
        except AssertionError:
            print(times)
            print(time0, time1, time_interval, ptnms)
            print('features',features)
            continue

        preds = config.classifier(solves, cmax_time) 

        loss = utils.compute_loss_on_train(config.criterion, labels, preds)
        try: 
            loss.backward()
        except RuntimeError:
            print(ptnms)
            print(times)
            continue
        config.optimizer.step()

    with torch.no_grad():
        
        train_res = utils.compute_loss_on_test(config.encoder, config.ode_func, config.classifier, config.args,
            config.data_obj["train_dataloader"], config.data_obj["n_train_batches"], 
            config.device, phase="train", AMT_index=-1-config.args.output_dim, latent_dim=config.args.latent_dim
            )

        validation_res = utils.compute_loss_on_test(config.encoder, config.ode_func, config.classifier, config.args,
            config.data_obj["val_dataloader"], config.data_obj["n_val_batches"], 
            config.device, phase="validate", AMT_index=-1-config.args.output_dim, latent_dim=config.args.latent_dim
            )
        
        train_loss = train_res["loss"] 
        validation_loss = validation_res["loss"]
        if validation_loss < config.args.best_rmse:
            torch.save({'encoder': config.encoder.state_dict(),
                        'ode': config.ode_func.state_dict(),
                        'classifier': config.classifier.state_dict(),
                        'args': config.args}, config.args.best_ckpt_path)
            config.args.best_rmse = validation_loss
            config.args.best_epochs = epoch
        torch.save({'encoder': config.encoder.state_dict(),
                        'ode': config.ode_func.state_dict(),
                        'classifier': config.classifier.state_dict(),
                        'args': config.args}, config.args.ckpt_path)

        message = """
        Epoch {:04d} | Training loss {:.6f} | Training R2 {:.6f} | Validation loss {:.6f} | Validation R2 {:.6f}
        Best loss {:.6f} | Best epoch {:04d}
        """.format(epoch, train_loss, train_res["r2"], validation_loss, validation_res["r2"],  config.args.best_rmse, config.args.best_epochs)
        config.logger.info(message)
        config.args.best_rmse = float(config.args.best_rmse)
    

