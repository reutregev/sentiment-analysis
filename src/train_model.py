import os
import subprocess

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification

from src.log_metrics import MetricsLogger
from src.logger import logger
from src.model_params import NUM_EPOCHS, LR, BATCH_SIZE, TOLERANCE, MIN_EPS
from src.utils import calc_running_time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TrainingModel:

    def __init__(self,
                 model_type: str,
                 num_labels: int,
                 batch_size: int = BATCH_SIZE,
                 n_epochs: int = NUM_EPOCHS,
                 lr: float = LR,
                 tolerance: int = TOLERANCE,
                 min_eps: float = MIN_EPS,
                 tensorboard=True
                 ):

        self.model = AutoModelForSequenceClassification.from_pretrained(model_type,
                                                                        num_labels=num_labels,
                                                                        output_attentions=False,
                                                                        output_hidden_states=False).to(device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.min_eps = min_eps
        self.val_loss_incr_counter = 0
        self.num_training_steps = self.n_epochs * self.batch_size
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=self.num_training_steps)
        self.metrics_logger = MetricsLogger(tensorboard)

    @calc_running_time
    def train(self, train_data: DataLoader, val_data: DataLoader, model_output_dir: str):
        train_size = len(train_data.dataset)
        val_size = len(val_data.dataset)

        logger.info(f"Started training. Num samples in train: {train_size} ; Num epochs: {self.n_epochs}")

        for epoch in tqdm(range(self.n_epochs)):
            curr_train_loss = 0
            curr_train_acc = 0
            curr_val_loss = 0
            curr_val_acc = 0

            self.model.train()

            for train_input_ids, train_attention_mask, train_labels in train_data:
                train_input_ids = train_input_ids.to(device)
                train_attention_mask = train_attention_mask.to(device)
                train_labels = train_labels.to(device)

                self.optimizer.zero_grad()
                train_outputs = self.model(train_input_ids, train_attention_mask, labels=train_labels)
                train_loss, train_logits = train_outputs[0], train_outputs[1]
                curr_train_loss += train_loss.item() / train_size

                curr_train_acc += (train_logits.argmax(dim=1) == train_labels).sum().item() / train_size

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                train_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            self.metrics_logger.add_metrics({"train_loss": curr_train_loss,
                                             "train_accuracy": curr_train_acc})

            self.model.eval()
            with torch.no_grad():
                for val_input_ids, val_attention_mask, val_labels in val_data:
                    val_input_ids = val_input_ids.to(device)
                    val_attention_mask = val_attention_mask.to(device)
                    val_labels = val_labels.to(device)

                    val_outputs = self.model(val_input_ids, val_attention_mask, labels=val_labels)
                    val_loss, val_logits = val_outputs[0], val_outputs[1].detach()
                    curr_val_loss += val_loss.item() / val_size

                    curr_val_acc += (val_logits.argmax(dim=1) == val_labels).sum().item() / val_size

            prev_val_loss = self.metrics_logger.get("val_loss")

            self.metrics_logger.add_metrics({"val_loss": curr_val_loss, "val_accuracy": curr_val_acc})
            self.metrics_logger.print_latest_epoch()
            self.metrics_logger.increase_epoch()

            if self.early_stopping(curr_val_loss, prev_val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(f"End training")
        self.save(model_output_dir)

        self.metrics_logger.run_tensorboard()

    def save(self, dir_path: str):
        logger.info(f"Saving model in {dir_path}")
        self.model.save_pretrained(os.path.join(dir_path, "trained_model1.pt"))

    def early_stopping(self, current_validation_loss: float, prev_val_loss: float) -> bool:
        if prev_val_loss - current_validation_loss < self.min_eps:
            self.val_loss_incr_counter += 1
            if self.val_loss_incr_counter >= self.tolerance:
                return True
        else:
            self.val_loss_incr_counter = 0
            return False
