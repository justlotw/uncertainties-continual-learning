import os
import sys
from shutil import copy2, rmtree

import torch
import torch.nn as nn
import torch.optim as optim

from time import time 
import numpy as np
import pandas as pd
from collections import OrderedDict

from models.convolution_models import get_conv_model
from plugins import ModelSavePlugin

from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import TextLogger


class TrainAvalancheExperiment:
    def __init__(self, train_stream):
        self.train_stream = train_stream

    def train(self, model, model_num, plugins, output_folder, n_epochs, batch_size, model_save_interval, lr=3e-4, seed=42, num_workers=4, **kwargs):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        RNGManager.set_random_seeds(seed)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Save model before training starts
        folder_path = os.path.join(output_folder, f"experience_{-1}", f"epoch_{0}", f"step_{0}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(model, os.path.join(folder_path, f"model_{model_num}.pt"))
        # model_scripted = torch.jit.script(model)
        # torch.jit.save(model_scripted, os.path.join(folder_path, f"model_{model_num}.pt"))

        plugins = plugins + [ModelSavePlugin(model, model_num, output_folder, model_save_interval),
                             EvaluationPlugin(accuracy_metrics(epoch=True, experience=True), 
                                              loss_metrics(epoch=True, experience=True),
                                              timing_metrics(epoch=True, experience=True),
                                              loggers=[TextLogger(open(os.path.join(output_folder, f"training_output_{model_num}.txt"), 'a'))])]

        cl_strategy = SupervisedTemplate(model, optimizer, criterion,
                               train_mb_size=batch_size, train_epochs=n_epochs,
                               plugins=plugins, device=device,
                                **kwargs)
        with open(os.path.join(output_folder, f"training_output_{model_num}.txt"), 'a') as f:
            print('Training on device:', cl_strategy.device, end='\n\n', file=f)
        cl_strategy.train(self.train_stream, eval_streams=[], num_workers=num_workers)

class TrainExperiment:
    def __init__(self, train_dataloaders):
        self.train_dataloaders = train_dataloaders
        self.num_experiences = len(train_dataloaders)
    
    def train(self, model, model_num, output_folder, n_epochs, model_save_interval=None, lr=3e-4, seed=42, verbose=True):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        torch.manual_seed(seed)

        self.ckpt_folder = output_folder
        if not os.path.exists(self.ckpt_folder):
            os.makedirs(self.ckpt_folder)   
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if verbose:
            np.set_printoptions(precision=4)
            print(f"Training on device: {self.device}")

        self.model = model
        self.model_num = model_num
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = n_epochs
        
        if model_save_interval is None:
            self.model_save_interval = n_epochs // 5
        else:
            self.model_save_interval = model_save_interval
            
        self.model.to(self.device)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self._save_model(-1, 0, 0)

        results = pd.DataFrame(columns=["Experience", "Epoch", "Train_Loss", "Train_Accuracy", "Time"])

        for experience_num, train_dataloader in enumerate(self.train_dataloaders):
            for epoch_num in range(n_epochs):
                start_time = time()
                loss, accuracy = self._training_loop(train_dataloader, experience_num, epoch_num, verbose=verbose)
                end_time = time()
                if verbose:
                    print(f"Experience: {experience_num}, Epoch: {epoch_num}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Time: {end_time - start_time:.2f}", flush=True)
                results.loc[len(results.index)] = [experience_num, epoch_num, loss, accuracy, end_time - start_time]
            print()

        results.to_csv(os.path.join(output_folder, f"training_results_{self.model_num}.csv"), index=False)

    def _save_model(self, experience_num, epoch_num, step_num):
        folder_path = os.path.join(self.ckpt_folder, f"experience_{experience_num}", f"epoch_{epoch_num}", f"step_{step_num}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.model, os.path.join(folder_path, f"model_{self.model_num}.pt"))
        # model_scripted = torch.jit.script(self.model)
        # torch.jit.save(model_scripted, os.path.join(folder_path, f"model_{self.model_num}.pt"))
    
    def _training_loop(self, dataloader, experience_num, epoch_num, verbose):
        device = self.device
        epoch_0 = epoch_num == 0
        running_loss = 0
        correct = 0
        total = 0

        for step_num, data in enumerate(dataloader):
            inputs, labels = data
            labels = labels.squeeze()
            inputs, labels = inputs.to(device), labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if epoch_0 and (step_num in [0, 2, 4] or step_num % 50 == 0):
                if verbose:
                    print(f"\tExperience: {experience_num}, Step: {step_num}, Loss: {loss.item():.4f}", flush=True)
                self._save_model(experience_num, epoch_num, step_num)
                

        loss = running_loss / total
        accuracy = correct / total

        if epoch_num % self.model_save_interval == 0 or epoch_num == self.num_epochs - 1:
            self._save_model(experience_num, epoch_num, step_num)

        return loss, accuracy
    

class TestExperiment:
    def __init__(self, test_dataloaders):
        self.test_dataloaders = test_dataloaders
        self.num_experiences = len(test_dataloaders)
        self.eps = 1e-10
        self.nll = nn.NLLLoss()
        self.n_bins = 10

    def test(self, ckpt_folder, results_folder, method, overwrite=False, verbose=True, **kwargs):
        if os.path.exists(results_folder) and not overwrite:
            raise Exception("Output folder already exists")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)     

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ckpt_folder = ckpt_folder
        self.verbose = verbose
        
        if method == "mc_dropout":
            try:
                self._testing_loop_mc_dropout(**kwargs)
            except Exception as e:
                self.results.to_csv(os.path.join(results_folder, "testing_results.csv"), index=False)
                raise e
        elif method == "ensemble":
            try:
                self._testing_loop_ensemble(**kwargs)
            except Exception as e:
                self.results.to_csv(os.path.join(results_folder, "testing_results.csv"), index=False)
                raise e
        elif method == "baseline":
            try:
                self._testing_loop_baseline(**kwargs)
            except Exception as e:
                self.results.to_csv(os.path.join(results_folder, "testing_results.csv"), index=False)
                raise e
        else:
            raise Exception("Invalid method. Must be mc_dropout or ensemble or baseline")

        self.results.to_csv(os.path.join(results_folder, "results.csv"), index=False)

    def calc_loss(self, outputs, labels):
        """
        Outputs are the logits (batch_size, num_classes)
        """
        log_outputs = torch.log(outputs + self.eps)
        return self.nll(log_outputs, labels).item()
    
    def calc_n_correct(self, predictions, labels):
        """
        Predictions are the max of the logits (batch_size)
        """
        return (predictions == labels).sum().item()
    
    def calc_variance(self, ensem_outputs):
        """
        Variance across samples. Ensem_outputs is (n_samples, batch_size, num_classes)
        """
        variance = torch.var(ensem_outputs, dim=0)
        return variance.sum().item()
    
    def calc_entropy(self, outputs):
        """
        Outputs are the logits (batch_size, num_classes)
        """
        entropy = -torch.sum(outputs * torch.log2(outputs + self.eps), dim=1)
        return entropy # (batch_size)
    
    def calc_uncertainty(self, entropy, ensem_outputs):
        """
        Ensem_outputs is (n_samples, batch_size, num_classes)

        Returns epistemic and aleatoric uncertainty
        """
        # Aleatoric uncertainty
        aleatoric = torch.mean(torch.sum(-ensem_outputs * torch.log2(ensem_outputs + self.eps), dim=2), dim=0) # (batch_size)
        # Epistemic uncertainty
        epistemic = entropy - aleatoric
        return epistemic.sum().item(), aleatoric.sum().item()
    
    def calc_ece(self, confidence, predictions, labels, n_bins=10):
        """
        Expected Calibration Error

        Confidence is (batch_size), represents the confidence of the max prediction
        Predictions is (batch_size)
        Labels is (batch_size)

        Returns the average confidence of each bin, the n_correct of each bin, and the number of samples in each bin
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_confidence = np.zeros(n_bins)
        correct_bin = np.zeros(n_bins)
        total_bin = np.zeros(n_bins)

        bins = np.digitize(confidence, bin_boundaries, right=True) # (batch_size), bin number for each sample
        for i in range(n_bins):
            bin_idx = bins == i + 1
            bin_confidence[i] += confidence[bin_idx].sum()
            bin_correct = predictions[bin_idx] == labels[bin_idx]
            correct_bin[i] += bin_correct.sum()
            total_bin[i] += bin_idx.sum()

        return bin_confidence, correct_bin, total_bin
    
    def calc_sce(self, outputs, labels, n_bins=10):
        """
        Static Calibration Error

        Outputs is (batch_size, num_classes)
        Labels is (batch_size)
        """
        num_classes = outputs.shape[1]
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        bin_confidence = np.zeros((n_bins, num_classes))
        correct_bin = np.zeros((n_bins, num_classes))
        total_bin = np.zeros((n_bins, num_classes))

        bins = np.digitize(outputs, bin_boundaries, right=True) # (batch_size, num_classes), bin number for each sample
        correct_preds = bins * nn.functional.one_hot(labels, num_classes=num_classes).numpy()
        for i in range(n_bins):
            bins_idx = bins == i + 1 
            bin_confidence[i] += (outputs * bins_idx).numpy().sum(axis=0)
            correct_bin[i] += (correct_preds == (i+1)).sum(axis=0)
            total_bin[i] += bins_idx.sum(axis=0)       
        return bin_confidence, correct_bin, total_bin
    
    def calc_uce(self, entropy, predictions, labels, n_bins=10):
        """
        Uncertainty Calibration Error

        Entropy is (batch_size), it should be normalized
        Predictions is (batch_size)
        Labels is (batch_size)
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_entropy = np.zeros(n_bins)
        error_bin = np.zeros(n_bins)
        total_bin = np.zeros(n_bins)

        bins = np.digitize(entropy, bin_boundaries, right=True)
        for i in range(n_bins):
            bin_idx = bins == i + 1
            bin_entropy[i] += entropy[bin_idx].sum()
            bin_error = predictions[bin_idx] != labels[bin_idx]
            error_bin[i] += bin_error.sum()
            total_bin[i] += bin_idx.sum()

        return bin_entropy, error_bin, total_bin
    
    def calc_signal_to_noise(self, logits):
        """
        Logits is (n_samples, batch_size, num_classes)
        """
        mean = torch.mean(torch.abs(logits), dim=0)
        variance = torch.var(logits, dim=0)
        snr = mean / (variance + self.eps)
        snr = snr.mean(dim=1)
        return snr.sum().item()

    def _testing_loop_baseline(self):
        softmax = nn.Softmax(dim=1)
        self.results = pd.DataFrame(columns=["Train_Experience", "Epoch", 
                                             "Step", "Test_Experience", 
                                             "Loss", "Accuracy", 
                                             "Entropy", 
                                             "ECE", "SCE", "UCE",
                                             "Time"])
        
        experiences = os.listdir(self.ckpt_folder)
        for experience in experiences:
            if not experience.startswith("experience_"):
                    continue
            experience_num = int(experience.split("_")[1])
            epoch = os.listdir(os.path.join(self.ckpt_folder, experience))
            for epoch_num in epoch:
                e_num = int(epoch_num.split("_")[1])
                step = os.listdir(os.path.join(self.ckpt_folder, experience, epoch_num))
                for step_num in step:
                    s_num = int(step_num.split("_")[1])
                    if not os.path.exists(os.path.join(self.ckpt_folder, experience, epoch_num, step_num, "model_0.pt")): 
                        raise Exception(f"Model does not exist for experience:{experience_num}, epoch: {e_num}, step: {s_num}")
                    for test_experience, dataloader in enumerate(self.test_dataloaders):
                        start_time = time()
                        # model = torch.jit.load(os.path.join(self.ckpt_folder, experience, epoch_num, step_num, "model_0.pt"))
                        model = torch.load(os.path.join(self.ckpt_folder, experience, epoch_num, step_num, "model_0.pt"))

                        # HACK FOR NOW
                        if isinstance(model, OrderedDict):
                            state_dict = model
                            model = get_conv_model('efficientnet-b0', num_classes=50)
                            model.load_state_dict(state_dict)

                        model.to(self.device)
                        model.eval()

                        running_loss = 0
                        correct = 0
                        accumulated_entropy = 0

                        bin_confidence_ece = np.zeros(self.n_bins)
                        correct_bin_ece = np.zeros(self.n_bins)
                        total_bin_ece = np.zeros(self.n_bins)

                        bin_confidence_sce = None 
                        correct_bin_sce = None
                        total_bin_sce = None

                        bin_confidence_uce = np.zeros(self.n_bins)
                        error_bin_uce = np.zeros(self.n_bins)
                        total_bin_uce = np.zeros(self.n_bins)

                        total = 0

                        with torch.no_grad():
                            for data in dataloader:
                                if len(data) == 3:
                                    inputs, labels, _ = data
                                else:
                                    inputs, labels = data
                                labels = labels.squeeze()
                                inputs = inputs.to(self.device)

                                mean = softmax(model(inputs)).cpu()  # (batch_size, num_classes)

                                if bin_confidence_sce is None:
                                    num_classes = mean.shape[1]
                                    bin_confidence_sce = np.zeros((self.n_bins, num_classes))
                                    correct_bin_sce = np.zeros((self.n_bins, num_classes))
                                    total_bin_sce = np.zeros((self.n_bins, num_classes))

                                confidence, predicted = torch.max(mean.data, 1)
                                confidence = confidence.cpu().numpy()

                                running_loss += self.calc_loss(mean, labels)
                                correct += self.calc_n_correct(predicted, labels)
                                entropy = self.calc_entropy(mean)
                                accumulated_entropy += entropy.sum().item()

                                conf, c_bin, n_bin = self.calc_ece(confidence, predicted, labels, n_bins=self.n_bins)
                                bin_confidence_ece += conf
                                correct_bin_ece += c_bin
                                total_bin_ece += n_bin

                                conf, c_bin, n_bin = self.calc_sce(mean, labels, n_bins=self.n_bins)
                                bin_confidence_sce += conf
                                correct_bin_sce += c_bin
                                total_bin_sce += n_bin

                                normalized_entropy = entropy / torch.log2(torch.tensor(num_classes).float())
                                conf, c_bin, n_bin = self.calc_uce(normalized_entropy, predicted, labels, n_bins=self.n_bins)
                                bin_confidence_uce += conf
                                error_bin_uce += c_bin
                                total_bin_uce += n_bin

                                total += labels.size(0)

                        accuracy = correct / total
                        loss = running_loss / total
                        entropy = accumulated_entropy / total

                        bin_confidence_ece = bin_confidence_ece / (total_bin_ece + self.eps)
                        ece = np.sum(np.abs(bin_confidence_ece - correct_bin_ece / (total_bin_ece + self.eps)) * total_bin_ece) / total
                        assert correct_bin_ece.sum() / total_bin_ece.sum() == accuracy

                        bin_confidence_sce = bin_confidence_sce / (total_bin_sce + self.eps)
                        sce = (np.abs((bin_confidence_sce - correct_bin_sce / (total_bin_sce + self.eps))) * total_bin_sce / total).sum(axis=0).mean()

                        bin_confidence_uce = bin_confidence_uce / (total_bin_uce + self.eps)
                        uce = np.sum(np.abs(bin_confidence_uce - error_bin_uce / (total_bin_uce + self.eps)) * total_bin_uce) / total

                        end_time = time()
                        if self.verbose:
                            print(f"Train Experience: {experience_num}, Epoch: {e_num}", 
                                  f"Step: {s_num}, Test Experience: {test_experience}",
                                  f"Accuracy: {accuracy:4f}, Loss: {loss:4f}",
                                  f"Entropy: {entropy:.4f}",
                                  f"ECE: {ece:.4f}, SCE: {sce:.4f}, UCE: {uce:.4f}", 
                                  f"Time: {end_time - start_time:2f}",
                                  flush=True, sep=', ')
                        self.results.loc[len(self.results)] = [experience_num, e_num, 
                                                               s_num, test_experience, 
                                                               loss, accuracy,
                                                               entropy, 
                                                               ece, sce, uce,  
                                                               end_time - start_time]
            print()


    def _testing_loop_mc_dropout(self, n_samples=10):
        softmax = nn.Softmax(dim=2)
        self.results = pd.DataFrame(columns=["Train_Experience", "Epoch", 
                                             "Step", "Test_Experience", 
                                             "Loss", "Accuracy", 
                                             "Variance", "Entropy", 
                                             "Mutual_Info", "Aleatoric_Uncertainty",
                                             "SNR_Logit", "SNR_Softmax",
                                             "ECE", "SCE", "UCE",
                                             "Time"])

        experiences = os.listdir(self.ckpt_folder)
        for experience in experiences:
            if not experience.startswith("experience_"):
                continue
            experience_num = int(experience.split("_")[1])
            epoch = os.listdir(os.path.join(self.ckpt_folder, experience))
            for epoch_num in epoch:
                e_num = int(epoch_num.split("_")[1])
                step = os.listdir(os.path.join(self.ckpt_folder, experience, epoch_num))
                for step_num in step:
                    s_num = int(step_num.split("_")[1])
                    if not os.path.exists(os.path.join(self.ckpt_folder, experience, epoch_num, step_num, "model_0.pt")): 
                        raise Exception(f"Model does not exist for experience:{experience_num}, epoch: {e_num}, step: {s_num}")
                    for test_experience, dataloader in enumerate(self.test_dataloaders):
                        start_time = time()
                        # model = torch.jit.load(os.path.join(self.ckpt_folder, experience, epoch_num, step_num, "model_0.pt"))
                        model = torch.load(os.path.join(self.ckpt_folder, experience, epoch_num, step_num, "model_0.pt"))
                        
                        # HACK FOR NOW
                        if isinstance(model, OrderedDict):
                            state_dict = model
                            model = get_conv_model('efficientnet-b0', num_classes=50)
                            model.load_state_dict(state_dict)

                        model.to(self.device)
                        model.eval()

                        for m in model.modules():
                            if m.__class__.__name__ == 'Dropout':
                                m.training = True

                        # for m in model.modules():
                        #     if m.original_name.startswith('Dropout'):
                        #         m.training = True
                        
                        running_loss = 0
                        correct = 0
                        accumulated_variance = 0
                        accumulated_entropy = 0
                        accumulated_mutual_info = 0
                        accumulated_aleatoric = 0
                        accumulated_snr_logit = 0
                        accumulated_snr_softmax = 0

                        bin_confidence_ece = np.zeros(self.n_bins)
                        correct_bin_ece = np.zeros(self.n_bins)
                        total_bin_ece = np.zeros(self.n_bins)

                        bin_confidence_sce = None 
                        correct_bin_sce = None
                        total_bin_sce = None

                        bin_confidence_uce = np.zeros(self.n_bins)
                        error_bin_uce = np.zeros(self.n_bins)
                        total_bin_uce = np.zeros(self.n_bins)

                        total = 0

                        with torch.no_grad():
                            for data in dataloader:
                                if len(data) == 3:
                                    inputs, labels, _ = data
                                else:
                                    inputs, labels = data
                                labels = labels.squeeze()
                                inputs = inputs.to(self.device)

                                logits = []
                                ensem_outputs = []
                                for _ in range(n_samples):
                                    outputs = model(inputs)
                                    logits.append(outputs)
                                logits = torch.stack(logits)  # (n_samples, batch_size, num_classes)
                                ensem_outputs = softmax(logits).cpu()  # (n_samples, batch_size, num_classes)

                                if bin_confidence_sce is None:
                                    num_classes = ensem_outputs.shape[2]
                                    bin_confidence_sce = np.zeros((self.n_bins, num_classes))
                                    correct_bin_sce = np.zeros((self.n_bins, num_classes))
                                    total_bin_sce = np.zeros((self.n_bins, num_classes))

                                mean = torch.mean(ensem_outputs, dim=0) # (batch_size, num_classes)
                                confidence, predicted = torch.max(mean.data, 1)
                                confidence = confidence.cpu().numpy()

                                running_loss += self.calc_loss(mean, labels)
                                correct += self.calc_n_correct(predicted, labels)
                                accumulated_variance += self.calc_variance(ensem_outputs)
                                entropy = self.calc_entropy(mean)
                                accumulated_entropy += entropy.sum().item()
                                mutual_info, aleatoric = self.calc_uncertainty(entropy, ensem_outputs)
                                accumulated_mutual_info += mutual_info  
                                accumulated_aleatoric += aleatoric
                                accumulated_snr_logit += self.calc_signal_to_noise(logits)
                                accumulated_snr_softmax += self.calc_signal_to_noise(ensem_outputs)

                                conf, c_bin, n_bin = self.calc_ece(confidence, predicted, labels, n_bins=self.n_bins)
                                bin_confidence_ece += conf
                                correct_bin_ece += c_bin
                                total_bin_ece += n_bin

                                conf, c_bin, n_bin = self.calc_sce(mean, labels, n_bins=self.n_bins)
                                bin_confidence_sce += conf
                                correct_bin_sce += c_bin
                                total_bin_sce += n_bin

                                normalized_entropy = entropy / torch.log2(torch.tensor(num_classes).float())
                                conf, c_bin, n_bin = self.calc_uce(normalized_entropy, predicted, labels, n_bins=self.n_bins)
                                bin_confidence_uce += conf
                                error_bin_uce += c_bin
                                total_bin_uce += n_bin

                                total += labels.size(0)

                        accuracy = correct / total
                        loss = running_loss / total
                        variance = accumulated_variance / total
                        entropy = accumulated_entropy / total
                        mutual_info = accumulated_mutual_info / total
                        aleatoric = accumulated_aleatoric / total
                        snr_logit = accumulated_snr_logit / total
                        snr_softmax = accumulated_snr_softmax / total

                        bin_confidence_ece = bin_confidence_ece / (total_bin_ece + self.eps)
                        ece = np.sum(np.abs(bin_confidence_ece - correct_bin_ece / (total_bin_ece + self.eps)) * total_bin_ece) / total
                        assert correct_bin_ece.sum() / total_bin_ece.sum() == accuracy

                        bin_confidence_sce = bin_confidence_sce / (total_bin_sce + self.eps)
                        sce = (np.abs((bin_confidence_sce - correct_bin_sce / (total_bin_sce + self.eps))) * total_bin_sce / total).sum(axis=0).mean()

                        bin_confidence_uce = bin_confidence_uce / (total_bin_uce + self.eps)
                        uce = np.sum(np.abs(bin_confidence_uce - error_bin_uce / (total_bin_uce + self.eps)) * total_bin_uce) / total

                        end_time = time()
                        if self.verbose:
                            print(f"Train Experience: {experience_num}, Epoch: {e_num}", 
                                  f"Step: {s_num}, Test Experience: {test_experience}",
                                  f"Accuracy: {accuracy:4f}, Loss: {loss:4f}",
                                  f"Variance: {variance:4f}, Entropy: {entropy:.4f}",
                                  f"Mutual Info: {mutual_info:4f}, Aleatoric Uncertainty: {aleatoric:.4f}", 
                                  f"SNR_Logit: {snr_logit:.4f}, SNR_Softmax: {snr_softmax:.1f}",
                                  f"ECE: {ece:.4f}, SCE: {sce:.4f}, UCE: {uce:.4f}", 
                                  f"Time: {end_time - start_time:2f}",
                                  flush=True, sep=', ')
                        self.results.loc[len(self.results)] = [experience_num, e_num, 
                                                               s_num, test_experience, 
                                                               loss, accuracy,
                                                               variance, entropy, 
                                                               mutual_info, aleatoric, 
                                                               snr_logit, snr_softmax,
                                                               ece, sce, uce, 
                                                               end_time - start_time]
            print()

    def _testing_loop_ensemble(self):       
        softmax = nn.Softmax(dim=2)             
        self.results = pd.DataFrame(columns=["Train_Experience", "Epoch", 
                                             "Step", "Test_Experience", 
                                             "Loss", "Accuracy", 
                                             "Variance", "Entropy", 
                                             "Mutual_Info", "Aleatoric_Uncertainty",
                                             "SNR_Logit", "SNR_Softmax",
                                             "ECE", "SCE", "UCE",
                                             "Time"])
        
        experiences = os.listdir(self.ckpt_folder)
        for experience in experiences:
            if not experience.startswith("experience_"):
                continue
            experience_num = int(experience.split("_")[1])
            epoch = os.listdir(os.path.join(self.ckpt_folder, experience))
            for epoch_num in epoch:
                e_num = int(epoch_num.split("_")[1])
                step = os.listdir(os.path.join(self.ckpt_folder, experience, epoch_num))
                for step_num in step:
                    s_num = int(step_num.split("_")[1])
                    num_models = len(os.listdir(os.path.join(self.ckpt_folder, experience, epoch_num, step_num)))
                    if num_models == 0:
                        raise Exception(f"No models exist for experience:{experience_num}, epoch: {e_num}, step: {s_num}")
                    for test_experience, dataloader in enumerate(self.test_dataloaders):
                        start_time = time()

                        running_loss = 0
                        correct = 0
                        accumulated_variance = 0
                        accumulated_entropy = 0
                        accumulated_mutual_info = 0
                        accumulated_aleatoric = 0
                        accumulated_snr_logit = 0
                        accumulated_snr_softmax = 0

                        bin_confidence_ece = np.zeros(self.n_bins)
                        correct_bin_ece = np.zeros(self.n_bins)
                        total_bin_ece = np.zeros(self.n_bins)

                        bin_confidence_sce = None 
                        correct_bin_sce = None
                        total_bin_sce = None

                        bin_confidence_uce = np.zeros(self.n_bins)
                        error_bin_uce = np.zeros(self.n_bins)
                        total_bin_uce = np.zeros(self.n_bins)

                        total = 0

                        models = [] 
                        
                        for model_num in range(num_models):
                            model = torch.load(os.path.join(self.ckpt_folder, experience, epoch_num, step_num, f"model_{model_num}.pt"))
                            
                            # HACK FOR NOW
                            if isinstance(model, OrderedDict):
                                state_dict = model
                                model = get_conv_model('efficientnet-b0', num_classes=50)
                                model.load_state_dict(state_dict)
                                
                            # model = torch.jit.load(os.path.join(self.ckpt_folder, experience, epoch_num, step_num, f"model_{model_num}.pt"))
                            model.eval()
                            models.append(model)

                        with torch.no_grad():
                            for data in dataloader:
                                if len(data) == 3:
                                    inputs, labels, _ = data
                                else:
                                    inputs, labels = data
                                labels = labels.squeeze()
                                inputs = inputs.to(self.device)

                                logits = []
                                ensem_outputs = []
                                for model in models:
                                    model.to(self.device)
                                    outputs = model(inputs)
                                    logits.append(outputs)
                                    model.to("cpu")
                                    torch.cuda.empty_cache()
                                logits = torch.stack(logits)  # (n_samples, batch_size, num_classes)
                                ensem_outputs = softmax(logits).cpu()  # (n_samples, batch_size, num_classes)

                                if bin_confidence_sce is None:
                                    num_classes = ensem_outputs.shape[2]
                                    bin_confidence_sce = np.zeros((self.n_bins, num_classes))
                                    correct_bin_sce = np.zeros((self.n_bins, num_classes))
                                    total_bin_sce = np.zeros((self.n_bins, num_classes))

                                mean = torch.mean(ensem_outputs, dim=0) # (batch_size, num_classes)
                                confidence, predicted = torch.max(mean.data, 1)
                                confidence = confidence.cpu().numpy()

                                running_loss += self.calc_loss(mean, labels)
                                correct += self.calc_n_correct(predicted, labels)
                                accumulated_variance += self.calc_variance(ensem_outputs)
                                entropy = self.calc_entropy(mean)
                                accumulated_entropy += entropy.sum().item()
                                mutual_info, aleatoric = self.calc_uncertainty(entropy, ensem_outputs)
                                accumulated_mutual_info += mutual_info  
                                accumulated_aleatoric += aleatoric
                                accumulated_snr_logit += self.calc_signal_to_noise(logits)
                                accumulated_snr_softmax += self.calc_signal_to_noise(ensem_outputs)

                                conf, c_bin, n_bin = self.calc_ece(confidence, predicted, labels, n_bins=self.n_bins)
                                bin_confidence_ece += conf
                                correct_bin_ece += c_bin
                                total_bin_ece += n_bin

                                conf, c_bin, n_bin = self.calc_sce(mean, labels, n_bins=self.n_bins)
                                bin_confidence_sce += conf
                                correct_bin_sce += c_bin
                                total_bin_sce += n_bin

                                normalized_entropy = entropy / torch.log2(torch.tensor(num_classes).float())
                                conf, c_bin, n_bin = self.calc_uce(normalized_entropy, predicted, labels, n_bins=self.n_bins)
                                bin_confidence_uce += conf
                                error_bin_uce += c_bin
                                total_bin_uce += n_bin

                                total += labels.size(0)

                        accuracy = correct / total
                        loss = running_loss / total
                        variance = accumulated_variance / total
                        entropy = accumulated_entropy / total
                        mutual_info = accumulated_mutual_info / total
                        aleatoric = accumulated_aleatoric / total
                        snr_logit = accumulated_snr_logit / total
                        snr_softmax = accumulated_snr_softmax / total

                        bin_confidence_ece = bin_confidence_ece / (total_bin_ece + self.eps)
                        ece = np.sum(np.abs(bin_confidence_ece - correct_bin_ece / (total_bin_ece + self.eps)) * total_bin_ece) / total

                        assert correct_bin_ece.sum() / total_bin_ece.sum() == accuracy

                        bin_confidence_sce = bin_confidence_sce / (total_bin_sce + self.eps)
                        sce = (np.abs((bin_confidence_sce - correct_bin_sce / (total_bin_sce + self.eps))) * total_bin_sce / total).sum(axis=0).mean()

                        bin_confidence_uce = bin_confidence_uce / (total_bin_uce + self.eps)
                        uce = np.sum(np.abs(bin_confidence_uce - error_bin_uce / (total_bin_uce + self.eps)) * total_bin_uce) / total

                        end_time = time()
                        if self.verbose:
                            print(f"Train Experience: {experience_num}, Epoch: {e_num}", 
                                    f"Step: {s_num}, Test Experience: {test_experience}",
                                    f"Accuracy: {accuracy:4f}, Loss: {loss:4f}",
                                    f"Variance: {variance:4f}, Entropy: {entropy:.4f}",
                                    f"Mutual Info: {mutual_info:4f}, Aleatoric Uncertainty: {aleatoric:.4f}", 
                                    f"SNR_Logit: {snr_logit:.4f}, SNR_Softmax: {snr_softmax:.1f}",
                                    f"ECE: {ece:.4f}, SCE: {sce:.4f}, UCE: {uce:.4f}", 
                                    f"Time: {end_time - start_time:2f}",
                                    flush=True, sep=', ')
                        self.results.loc[len(self.results)] = [experience_num, e_num, 
                                                                s_num, test_experience, 
                                                                loss, accuracy,
                                                                variance, entropy, 
                                                                mutual_info, aleatoric, 
                                                                snr_logit, snr_softmax,
                                                                ece, sce, uce,  
                                                                end_time - start_time]
            print()
