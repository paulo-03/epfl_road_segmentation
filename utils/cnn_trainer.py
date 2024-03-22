from os import path, listdir

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torchvision.io import read_image
from torchvision.transforms import v2

from utils.cnn import CNN
from .performance_metrics import get_metrics


class CnnTrainer(CNN):
    def __init__(self, data_kwargs: dict, optimizer_kwargs: dict, num_epochs: int, model_name: str = "UNet",
                 model_saving_path: str = None, test_size: float = 0.25,  device: str = 'cpu') -> None:
        super().__init__(model_name, device)

        self.test_size = test_size
        self.num_epochs = num_epochs
        self.model_saving_path = model_saving_path
        self.batch_size = data_kwargs['batch_size']
        self.mean = data_kwargs['mean']
        self.std = data_kwargs['std']

        # Get loader from disk
        self.training_loader, self.validation_loader = self._get_data_loader_from_disk(**data_kwargs)

        # ensure that the function that got the dataloader correctly set the train and val set size
        assert self.train_set_size != 0, 'The train set size has not been properly set'
        assert self.val_set_size != 0, 'The val set size has not been properly set'

        self.training_batch_number = int(self.train_set_size / data_kwargs['batch_size'])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        self.criterion = nn.functional.binary_cross_entropy_with_logits
        self.schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=(len(self.training_loader.dataset) * num_epochs) // self.training_loader.batch_size
        )

    def restore_model(self, model_path: str) -> None:
        model_save = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_save['model_state_dict'])
        self.model = self.model.to(torch.device(self.device))  # ensure model is on correct device
        self.optimizer.load_state_dict(model_save['optimizer_state_dict'])
        self.schedular.load_state_dict(model_save['schedular_state_dict'])
        self.cur_epoch = model_save['epoch']
        self.train_loss_history = model_save['train_loss_history']
        self.train_acc_history = model_save['train_acc_history']
        self.lr_history = model_save['lr_history']
        self.val_loss_history = model_save['val_loss_history']
        self.val_acc_history = model_save['val_acc_history']
        self.train_prf1_p_history = model_save['train_prf1_p_history']
        self.train_prf1_n_history = model_save['train_prf1_n_history']
        self.val_prf1_p_history = model_save['val_prf1_p_history']
        self.val_prf1_n_history = model_save['val_prf1_n_history']

        # TODO uncomment that once file has been updated
        self.batch_size = model_save['batch_size']
        self.train_set_size = model_save['train_set_size']
        self.val_set_size = model_save['val_set_size']
        self.mean = model_save['mean']
        self.std = model_save['std']
        self.model_name = model_save['model_name']
        self.training_batch_number = model_save['training_batch_number']

    def fit(self) -> None:
        """Compute the training of the model"""

        # Start training
        for epoch in range(self.cur_epoch + 1, self.num_epochs + 1):
            self.cur_epoch += 1  # increment the current epoch counter

            print(f"Start Training Epoch {epoch}...")
            train_loss, train_acc, train_prf1_p, train_prf1_n, lrs = self._train_epoch()
            val_loss, val_acc, val_prf1_p, val_prf1_n = self._validate()

            print(
                f"- Average metrics: \n"
                f"\t\t- train loss={np.mean(train_loss):0.2e}, "
                f"train acc={np.mean(train_acc):0.3f}, "
                f"learning rate={np.mean(lrs):0.3e} \n"
                f"\t\t- val loss={val_loss:0.2e}, "
                f"val acc={val_acc:0.3f} \n"
                f"Finish Training Epoch {epoch} !\n"
            )

            # Store all metrics in array, to plot them at the end of training
            self.train_loss_history.extend(train_loss)
            self.train_acc_history.extend(train_acc)
            self.lr_history.extend(lrs)
            self.train_prf1_p_history.extend(train_prf1_p)
            self.train_prf1_n_history.extend(train_prf1_n)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            self.val_prf1_p_history.append(val_prf1_p)
            self.val_prf1_n_history.append(val_prf1_n)

            # Save the model
            if self.model_saving_path is not None:
                self._save_model()

        # Plot training curves
        self._plot_training_curves(self.num_epochs)

    @torch.no_grad()
    def _validate(self):
        """Compute the accuracy and loss for the validation set"""

        # Set the model in evaluation mode (turn-off the auto gradient computation, ...)
        self.model.eval()

        test_loss = 0
        accuracy = 0
        pos_precision = 0
        pos_recall = 0
        pos_f1 = 0
        neg_precision = 0
        neg_recall = 0
        neg_f1 = 0
        for data, target in self.validation_loader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)

            batch_size = len(data)

            test_loss += self.criterion(output, target).item() * batch_size

            # calculate the metrics
            formatted_output = output.clamp(0, 1).round()
            acc, (p_p, r_p, f1_p), (p_n, r_n, f1_n) = get_metrics(formatted_output, target)

            # multiply by batch_size, to compute weighted average over the whole val dataset
            accuracy += acc * batch_size
            pos_precision += p_p * batch_size
            pos_recall += r_p * batch_size
            pos_f1 += f1_p * batch_size
            neg_precision += p_n * batch_size
            neg_recall += r_n * batch_size
            neg_f1 += f1_n * batch_size

        test_loss /= self.val_set_size
        accuracy /= self.val_set_size
        pos_precision /= self.val_set_size
        pos_recall /= self.val_set_size
        pos_f1 /= self.val_set_size
        neg_precision /= self.val_set_size
        neg_recall /= self.val_set_size
        neg_f1 /= self.val_set_size

        return test_loss, accuracy, [pos_precision, pos_recall, pos_f1], [neg_precision, neg_recall, neg_f1]

    def _train_epoch(self):
        """Train one epoch"""

        # Set the model in training mode
        self.model.train()

        # Initiate array to store metrics evolution
        loss_history = []
        accuracy_history = []
        prf1_p_history = []
        prf1_n_history = []
        lr_history = []

        # Compute array only one per epoch
        batch_to_print = np.linspace(0, self.training_batch_number, 5).round()

        for batch_idx, (data, target) in enumerate(self.training_loader):
            # Move the data to the device
            data, target = data.to(self.device), target.to(self.device)
            # Zero the gradients
            self.optimizer.zero_grad()
            # Compute model output
            output = self.model(data)
            # Compute loss
            loss = self.criterion(output, target)
            # Backpropagation loss
            loss.backward()
            # Perform an optimizer step
            self.optimizer.step()
            # Perform a learning rate scheduler step (if schedular set)
            self.schedular.step()

            # Calculate batch metrics
            formatted_output = output.clamp(min=0, max=1).round()  # Map all values to 0 or 1
            accuracy, (p_p, r_p, f1_p), (p_n, r_n, f1_n) = get_metrics(formatted_output, target)

            # Store metrics
            loss_history.append(loss.item())
            accuracy_history.append(accuracy)
            lr_history.append(self.schedular.get_last_lr()[0])
            prf1_p_history.append([p_p, r_p, f1_p])
            prf1_n_history.append([p_n, r_n, f1_n])

            if batch_idx in batch_to_print:  # give a feedback 5 times per epoch
                print(
                    f"- Metrics of Batch number {batch_idx:03d}/{self.training_batch_number}: \n"
                    f"\t\t- loss={loss.item():0.2e}, "
                    f"acc={accuracy:0.3f}, "
                    f"lr={self.schedular.get_last_lr()[0]:0.3e} "
                )

        return loss_history, accuracy_history, prf1_p_history, prf1_n_history, lr_history

    def _save_model(self):
        """Save important variable used by the model"""
        state = {
            'epoch': self.cur_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'schedular_state_dict': self.schedular.state_dict(),
            'train_loss_history': self.train_loss_history,
            'train_acc_history': self.train_acc_history,
            'lr_history': self.lr_history,
            'val_loss_history': self.val_loss_history,
            'val_acc_history': self.val_acc_history,
            'train_prf1_p_history': self.train_prf1_p_history,
            'train_prf1_n_history': self.train_prf1_n_history,
            'val_prf1_p_history': self.val_prf1_p_history,
            'val_prf1_n_history': self.val_prf1_n_history,


            'batch_size': self.batch_size,
            'train_set_size': self.train_set_size,
            'val_set_size': self.val_set_size,
            'mean': self.mean,
            'std': self.std,
            'model_name': self.model_name,
            'training_batch_number': self.training_batch_number
        }

        torch.save(state, path.join(self.model_saving_path, f'training_save_epoch_{self.cur_epoch}.tar'))

    def _get_data_loader_from_disk(self, batch_size: int, mean: torch.Tensor, std: torch.Tensor, img_folder: str,
                                   gt_folder: str):
        """Helper function to load data into train and val dataloader directly from the disk"""

        class __RoadDataset(torch.utils.data.Dataset):
            """Class representing our custom Dataset"""

            def __init__(self):
                self.imgs = list(sorted(listdir(img_folder)))
                self.gts = list(sorted(listdir(gt_folder)))

                self.to_float = v2.ToDtype(torch.float32, scale=True)
                self.transform_norm = v2.Compose([
                    self.to_float,
                    v2.Normalize(mean, std)
                ])

            def __getitem__(self, idx: int):
                img_path = path.join(img_folder, self.imgs[idx])
                gt_path = path.join(gt_folder, self.gts[idx])

                img = read_image(img_path)
                gt = read_image(gt_path)

                img = self.transform_norm(img)
                gt = self.to_float(gt).round()

                return img, gt

            def __len__(self):
                return len(self.imgs)

        # Get the dataset
        dataset = __RoadDataset()
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)

        # Calculate the index at which to split
        split_idx = int(self.test_size * dataset_size)

        np.random.shuffle(indices)

        train_indices, val_indices = indices[split_idx:], indices[:split_idx]

        # Store the train and val indices length
        self.train_set_size = len(train_indices)
        self.val_set_size = len(val_indices)

        # Create the training and validation RandomSampler
        train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

        # Set the training and validation DataLoader
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                  pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, val_loader
