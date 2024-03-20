import os 
import torch 

from avalanche.core import BaseSGDPlugin

class ModelSavePlugin(BaseSGDPlugin):
    def __init__(self, model, model_num, save_dir, save_freq):
        super().__init__()
        self.model = model
        self.model_num = model_num
        self.save_dir = save_dir
        self.save_freq = save_freq

    def save_model(self, experience_num, epoch_num, step_num):
        folder_path = os.path.join(self.save_dir, f"experience_{experience_num}", f"epoch_{epoch_num}", f"step_{step_num}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.model, os.path.join(folder_path, f"model_{self.model_num}.pt"))
        # model_scripted = torch.jit.script(self.model)
        # torch.jit.save(model_scripted, os.path.join(folder_path, f"model_{self.model_num}.pt"))

    def after_training_epoch(self, strategy, **kwargs):
        if strategy.clock.train_exp_epochs <= 2 or strategy.clock.train_exp_epochs % self.save_freq == 0:
            self.save_model(strategy.clock.train_exp_counter, strategy.clock.train_exp_epochs, strategy.clock.train_epoch_iterations)

    def after_training_exp(self, strategy, **kwargs):
        self.save_model(strategy.clock.train_exp_counter, strategy.clock.train_exp_epochs, strategy.clock.train_epoch_iterations)