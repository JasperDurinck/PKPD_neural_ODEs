
import os
import pandas as pd
import torch
import utils
from model import *
from data_parse import parse_data

def move_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.to('cpu')
    elif isinstance(data, dict):
        return {key: move_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_cpu(item) for item in data]
    else:
        return data  # If it's not a tensor, return it as is

def predict(config):

    ###############################################################
    ## Main runnings
    ckpt_path = os.path.join(config.args.save, f"{config.args.model_name}_model_best.ckpt")
    eval_path = os.path.join(config.args.save, f"{config.args.model_name}_model_eval.csv")
    res_path = "rmse.csv"

    ########################################################################
    data_obj = parse_data(
        config.device, 
        phase="test", 
        dir=config.args.DATA_DIR,
        label_cols=config.args.label_cols,
        feature_cols= config.args.feature_cols,
        covariates=config.args.covariates_cols)
    input_dim = data_obj["input_dim"]
    if config.args.input_dim is None: config.args.input_dim = input_dim

    utils.load_model(ckpt_path, config.encoder, config.ode_func, config.classifier, config.device)

    ########################################################################
    ## Predict & Evaluate
    with torch.no_grad():
        test_res = utils.compute_loss_on_test(config.encoder, config.ode_func, config.classifier, config.args,
            data_obj["test_dataloader"], data_obj["n_test_batches"], 
            config.device, phase="test", latent_dim=config.args.latent_dim, AMT_index=-1-config.args.output_dim, labels_n=config.args.output_dim)
    test_res = move_to_cpu(test_res)
    eval_results = pd.DataFrame(test_res).drop(columns="loss")
    eval_results.to_csv(eval_path, index=False)

