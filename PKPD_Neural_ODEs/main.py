import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from data_parse import parse_data
from run_train import train_one_epoch
from data_split import split_dataset
from process_data import process_data
from run_predict import predict as _predict
from model import *

class PKPD_neural_ODE:
    def __init__(self, load_config=None, save_config='config'):
        self.save_config = save_config
        if load_config is not None:
            self.save_config = load_config
        self.args = Args(load_config,save_config)
        utils.makedirs("logs/")
        self.log_path = "logs/" + f"model_{self.args.model_name}.log"
        self.input_cmd = sys.argv
        self.input_cmd = " ".join(self.input_cmd)
        self.logger = utils.get_logger(logpath=self.log_path, filepath=os.path.abspath(__file__))
        self.logger.info(self.input_cmd)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device', self.device)

    def _config(self):
        self.args = Args()

    def _load_data(self, phase="train"):
        
        self.data_obj = parse_data(
            self.device, 
            phase=phase, 
            dir=self.args.DATA_DIR, 
            label_cols=self.args.label_cols, 
            feature_cols=self.args.feature_cols,
            covariates=self.args.covariates_cols
            )
        
        self.args.input_dim = self.data_obj["input_dim"]

    def _initilize_model(self):

        self.encoder = Encoder(
            input_dim=self.args.input_dim, 
            output_dim=2 * self.args.latent_dim, 
            hidden_dim=self.args.encoder_hidden_dim, 
            device=self.device
            ).to(device=self.device)
        
        self.ode_func = ODEFunc(
            input_dim=self.args.latent_dim, 
            hidden_dim=self.args.ode_func_hidden_dim
            ).to(device=self.device)
        
        self.classifier = Classifier(
            latent_dim=self.args.latent_dim, 
            output_dim=self.args.output_dim
            ).to(device=self.device)
    
    def _process_data(self):
        process_data(self.args)
        split_dataset(self.args.STORE_DATA, self.args, test_size=0.2, seed=42, output_dir="./data") 

    def train_model(self):

        if self.args.continue_train:
            utils.load_model(self.args.ckpt_path, self.encoder, self.ode_func, self.classifier, self.device)

        self.criterion = nn.MSELoss().to(device=self.device)
        self.params = (list(self.encoder.parameters()) + 
                list(self.ode_func.parameters()) + 
                list(self.classifier.parameters()))
        self.optimizer = optim.Adam(self.params, lr=self.args.lr, weight_decay=self.args.l2)

        for epoch_num in range(self.args.current_epoch, self.args.epochs):
            train_one_epoch(self)
            self.args.current_epoch+=1
            self.args.save_to_json(self.save_config)

    def predict(self): _predict(self)


class Args:
    def __init__(self, load_config=None, save_config='config.json'):
        # Default settings
        self.model_name = 'test1'
        self.save = './results/model_ckpt/'
        self.save_best = './results/model_ckpt/'
        self.seed = 42
        self.ckpt_path = os.path.join(self.save, f"{self.model_name}_model.ckpt")
        self.best_ckpt_path = os.path.join(self.save, f"{self.model_name}_model_best.ckpt")

        # Training settings
        self.best_rmse = 0x7fffffff
        self.best_epochs = 0
        self.current_epoch = 0
        self.lr = 0.001
        self.l2 = 0.1
        self.epochs = 10
        self.tol = 0.0001
        self.continue_train = False

        # Model settings
        self.n_labels = 2
        self.features = 8
        self.encoder_hidden_dim = 128 * self.n_labels 
        self.latent_dim = self.features * self.n_labels
        self.output_dim = self.n_labels
        self.ode_func_hidden_dim = 16
        self.input_dim = None  # load_data

        # Data split settings
        self.DSFQ_SETTINGS = {1: 1*48, 2: 2*48, 3: 48*3}
        self.AUGMENTATION_SETTINGS = {
            'interval': 48,
            'ranges': [2 ,4, 6],
            'offset_values': [0.1, 0.2, 0.3]
        }

        # Process data settings
        self.DATA_DIR = 'data/'
        self.GET_DATA = f"{self.DATA_DIR}sim_data.csv"
        self.STORE_DATA = f"{self.DATA_DIR}data.csv"
        self.MAX_CYCLES = 100
        self.PTNM_MAX_CHAR = 5
        self.MINIMAL_START_TIME = 0
        self.COLUMNS_TO_SELECT = ["STUD", "DSFQ", "PTNM", "CYCL", "AMT", "TIME", "TFDS", "DV1", "DV2", 'age', 'weight']
        self.COLUMNS_TO_RENAME = {"DV1": "PK_timeCourse_1", "DV2": "PK_timeCourse_2"}

        self.label_cols = ["PK_timeCourse_1", "PK_timeCourse_2"]
        self.feature_cols = ['TFDS', 'TIME', 'CYCL', 'AMT', "PK1_round1", "PK2_round1"]
        self.covariates_cols = ['age', 'weight']

        if load_config is not None: 
            self.load_from_json(load_config)
            self.continue_train = True
        else:
            self.save_to_json(save_config)

    def save_to_json(self, file_path):
        """Save configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def load_from_json(self, file_path):
        """Load configuration from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)


if __name__ == '__main__':

    PKPD_NODEs = PKPD_neural_ODE(load_config='configs/config_test.json')
    #PKPD_NODEs._process_data()
    PKPD_NODEs._load_data()
    PKPD_NODEs._initilize_model()
    #PKPD_NODEs.train_model()
    PKPD_NODEs.predict()






