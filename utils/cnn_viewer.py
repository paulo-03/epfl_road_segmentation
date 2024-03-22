import torch
from PIL.Image import Image
from torchvision import transforms

from utils.cnn import CNN
from utils.performance_metrics import get_performance_distribution


class CnnViewer(CNN):
    def __init__(self, model_path: str, model_name: str = 'UNet', device: str = 'cpu'):
        super().__init__(model_name, device)

        self.restore_model(model_path)

    def restore_model(self, model_path: str) -> None:
        model_save = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_save['model_state_dict'])
        self.model = self.model.to(torch.device(self.device))  # ensure model is on correct device
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

    @torch.no_grad()
    def _compute_pred_tensor(self, img_fold: list[Image], gt_fold: list[Image]):
        self.model.eval()

        to_float = transforms.ToTensor()

        gt_tensor = torch.stack([to_float(gt) for gt in gt_fold])
        gt_tensor.to(self.device)

        # Compute the predictions
        pred_list = self.predict(img_fold)
        pred_tensor = torch.stack([to_float(pred) for pred in pred_list])
        pred_tensor.to(self.device)

        return pred_tensor, gt_tensor

    def performance_distribution(self, img_folds: list[list[Image]], gt_folds: list[list[Image]]) -> list[list[float]]:
        """
        Function that performs a performance distribution of the model, by making prediction of k fold validation
        images.

        :param img_folds: The image folds list
        :param gt_folds: The groundtruth fold list

        :return: The performance metrics of each k_fold as a list
        """

        return get_performance_distribution(img_folds, gt_folds, self._compute_pred_tensor)
