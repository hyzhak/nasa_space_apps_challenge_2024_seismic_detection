from accelerate import Accelerator
from momentfm import MOMENTPipeline
import numpy as np
import os
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MomentTrainer:
    """
    Inspired by https://github.com/moment-timeseries-foundation-model/moment/blob/040df9a5091eb83201d8764acb96350a5c84f47e/tutorials/finetune_demo/classification.py
    """

    def __init__(self,
                 train_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 epochs: int = 5,
                 init_lr: float = 1e-6,
                 max_lr: float = 1e-4,
                 # 'reduction method for MOMENT embeddings, choose from mean or max'
                 reduction: str = 'mean',
                 mode: str = 'linear_probing',
                 n_channels: int = 1,
                 output_path: str = 'output',
                 num_class: int = 1,
                 lora=False):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.reduction = reduction
        self.mode = mode
        self.output_path = output_path

        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': 'classification',
                'n_channels': n_channels,
                'num_class': 2,
                'freeze_encoder': False if self.mode == 'full_finetuning' else True,
                'freeze_embedder': False if self.mode == 'full_finetuning' else True,
                'reduction': self.reduction,
                # Disable gradient checkpointing for finetuning or linear probing to
                # avoid warning as MOMENT encoder is frozen
                'enable_gradient_checkpointing': False if self.mode in ['full_finetuning', 'linear_probing'] else True,
            },
        )
        self.model.init()

        self.criterion = torch.nn.CrossEntropyLoss()

        if self.mode == 'full_finetuning':
            print('Encoder and embedder are trainable')
            if lora:
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=32,
                    target_modules=["q", "v"],
                    lora_dropout=0.05,
                )
                config_original = getattr(self.model, "config")
                # try to solve issue
                # File /opt/conda/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:589, in BaseTuner._get_tied_target_modules(self, model)
                #     587 tied_target_modules = []
                #     588 model_config = self.get_model_config(model)
                # --> 589 if model_config.get("tie_word_embeddings"):
                #     590     for target_module in self.targeted_module_names:
                #     591         if target_module in EMBEDDING_LAYER_NAMES:
                # AttributeError: 'NamespaceWithDefaults' object has no attribute 'get'
                self.model.config = vars(config_original)
                self.model = get_peft_model(self.model, lora_config)
                print('LoRA enabled')
                self.model.print_trainable_parameters()
                # and then reset the config because of error
                #     510 enc_in = self.patch_embedding(x_enc, mask=input_mask)
                #     512 n_patches = enc_in.shape[2]
                #     513 enc_in = enc_in.reshape(
                # --> 514     (batch_size * n_channels, n_patches, self.config.d_model)
                #     515 )
                #     517 patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
                #     518 attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
                #
                # AttributeError: 'dict' object has no attribute 'd_model'
                self.model.get_base_model().config = config_original

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr,
                                                                 total_steps=epochs * len(self.train_dataloader))

            # set up model ready for accelerate finetuning
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr,
                                                                 total_steps=self.epochs * len(self.train_dataloader))
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        performance = []
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            # self.log_file.write(f'Epoch {epoch + 1}/{self.epochs}\n')
            self.epoch = epoch + 1

            if self.mode == 'linear_probing':
                train_loss = self.train_epoch_lp()
                val_loss, val_accuracy = self.evaluate_epoch('val')

            elif self.mode == 'full_finetuning':
                train_loss = self.train_epoch_ft()
                val_loss, val_accuracy = self.evaluate_epoch('val')
            # # break after training SVM, only need one 'epoch'
            # elif self.args.mode == 'unsupervised_representation_learning':
            #     self.train_ul()
            #     break
            #
            # elif self.args.mode == 'svm':
            #     self.train_svm()
            #     break
            #
            else:
                raise ValueError(
                    'Invalid mode, please choose svm, linear_probing, full_finetuning, or unsupervised_representation_learning')

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                print('Saving best model')
                self.save_checkpoint()

            performance.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

        return performance

    def train_epoch_lp(self):
        '''
        Train only classification head
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch_x, batch_labels in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and
                                                                            torch.cuda.get_device_capability()[
                                                                                0] >= 8 else torch.float32):
                output = self.model(x_enc=batch_x, reduction=self.reduction)
                loss = self.criterion(output.logits, batch_labels)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        return avg_loss

    def train_epoch_ft(self):
        '''
        Train encoder and classification head (with accelerate enabled)
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch_x, batch_labels in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and
                                                                            torch.cuda.get_device_capability()[
                                                                                0] >= 8 else torch.float32):
                output = self.model(x_enc=batch_x, reduction=self.reduction)
                loss = self.criterion(output.logits, batch_labels)
                losses.append(loss.item())
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.scheduler.step()

        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        return avg_loss

    def evaluate_epoch(self, phase='val'):
        if phase == 'train':
            dataloader = self.train_dataloader
        elif phase == 'val':
            dataloader = self.val_dataloader
        # elif phase == 'test':
        #     dataloader = self.test_dataloader
        else:
            raise ValueError('Invalid phase, please choose val or test')

        self.model.eval()
        self.model.to(self.device)
        total_loss, total_correct = 0, 0

        with torch.no_grad():
            for batch_x, batch_labels in tqdm(dataloader):
                batch_x = batch_x.to(self.device).float()
                batch_labels = batch_labels.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and
                                                                                torch.cuda.get_device_capability()[
                                                                                    0] >= 8 else torch.float32):
                    output = self.model(x_enc=batch_x)
                    loss = self.criterion(output.logits, batch_labels)
                total_loss += loss.item()
                total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / len(dataloader)
        print(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}')
        return avg_loss, accuracy

    def save_checkpoint(self):
        os.makedirs(self.output_path, exist_ok=True)
        torch.save(
            # self.model.get_base_model().state_dict(),
            self.model.state_dict(),
            os.path.join(self.output_path, 'MOMENT_Classification.pth'),
            # weights_only=True,
        )
        print('Model saved at ', self.output_path)


def load_checkpoint(checkpoint_path: str):
    mode = 'full_finetuning'
    reduction = 'mean'
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification',
            'n_channels': 1,
            'num_class': 2,
            # 'freeze_encoder': False if mode == 'full_finetuning' else True,
            # 'freeze_embedder': False if mode == 'full_finetuning' else True,
            # 'reduction': reduction,
            # # Disable gradient checkpointing for finetuning or linear probing to
            # # avoid warning as MOMENT encoder is frozen
            # 'enable_gradient_checkpointing': False if mode in ['full_finetuning', 'linear_probing'] else True,
        },
    )
    model.init()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_path, 'MOMENT_Classification.pth')),
        # weights_only=True,
    )
    return model
