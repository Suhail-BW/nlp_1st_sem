import os
import time
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import DataLoader

from sentiment.dataset import ReviewsDataset, RandomReviewsDataset
from sentiment.model import RobertaClassifier


def load_config(cwd, config_path):
    OmegaConf.register_resolver("now", lambda pattern: time.strftime(pattern, time.localtime()))
    config = OmegaConf.load(str(config_path))
    config_cli = OmegaConf.from_cli()

    config = OmegaConf.merge(config, config_cli)
    config.original_cwd = str(cwd)

    checkpoint_dir = Path(config.checkpoint_dir) / config.method_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.yaml", "w+") as file:
        OmegaConf.save(config, file)
    return config


class Runner:
    def __init__(self, model, config):
        self.config = config
        data_dir = config.data_dir
        # Dataset
        assert config.embedding_model in ["bert", "roberta"]
        if config.embedding_model == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        # train_dataset = RandomReviewsDataset(
        #     [
        #         Path(data_dir) / f"review_train_data_{rating}.csv"
        #         for rating in range(1, 6)
        #     ],
        #     self.tokenizer,
        #     config)
        train_dataset = ReviewsDataset(Path(data_dir) / "review_train_data.csv", self.tokenizer, config)
        valid_dataset = ReviewsDataset(Path(data_dir) / "review_valid_data.csv", self.tokenizer, config)
        print(f"Dataset size: {len(train_dataset)} {len(valid_dataset)}")
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers)
        self.valid_loader = DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers)

        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        """
        self.pretrained_dir = "/opt/weights"
        self.method_name = "roberta_fc-True_pooler-True_transformer-False_d0.0_wd1e-5_lr1e-4"
        self.model.load_state_dict(
            torch.load(Path(self.pretrained_dir) / self.method_name / "best_vloss-1.0295_vacc-0.7004.pth")["model"]
        )
        """

        # Optimizer
        pos_weight = torch.tensor([5.5, 15.0, 9.0, 5.0, 2.5]).to(self.device)
        # pos_weight = torch.tensor([
        #     4.0, 4.0 * config.class_balance,
        #     4.0 * config.class_balance,
        #     4.0 * config.class_balance, 4.0,
        # ]).to(self.device)
        self.criterion_rating = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        pos_weight = torch.tensor([2.0, 4.0, 1.0]).to(self.device)
        # pos_weight = torch.tensor([2.0, 4.0, 2.0]).to(self.device)
        self.criterion_sentiment = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # pos_weight = torch.tensor([1.25]).to(self.device)
        self.criterion_strength = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
        )

        # Checkpoint
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.method_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_value = 0.0
        self.best_checkpoint = None

    def compute_accuracy(self, outputs, targets):
        _, output_max_indices = torch.max(outputs.data, dim=1)
        _, target_max_indices = torch.max(targets.data, dim=1)
        count_correct = (output_max_indices == target_max_indices).sum().item()

        target_max_indices_p1 = torch.clamp(target_max_indices + 1, min=0, max=4)
        target_max_indices_m1 = torch.clamp(target_max_indices - 1, min=0, max=4)
        mask_count_close = ((output_max_indices >= target_max_indices_m1) & (output_max_indices <= target_max_indices_p1))
        count_close = mask_count_close.sum().item()

        return count_correct, count_close

    def compute_accuracy_details(self, outputs, targets):
        acc_correct, acc_close = self.compute_accuracy(outputs, targets)

        acc_by_rating = dict()
        _, output_max_indices = torch.max(outputs.data, dim=1)
        _, target_max_indices = torch.max(targets.data, dim=1)
        mask = (output_max_indices == target_max_indices)
        for i in range(0, 5):
            mask_target = (target_max_indices == i)
            mask_i = (mask & mask_target)
            acc_by_rating[i + 1] = [mask_i.sum().item(), mask_target.sum().item()]

        return acc_correct, acc_close, acc_by_rating

    def compute_accuracy_other(self, sentiments, sentiment_targets, strengths, strength_targets):
        _, output_max_indices = torch.max(sentiments.data, dim=1)
        _, target_max_indices = torch.max(sentiment_targets.data, dim=1)
        count_sentiment = (output_max_indices == target_max_indices).sum().item()

        strength_outputs = (strengths > 0.5)
        count_strength = (strength_outputs == strength_targets).sum().item()
        return count_sentiment, count_strength

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        step = 0
        print("Train: " + "." * 20)

        for step, batch in enumerate(self.train_loader, start=1):
            ids = batch["ids"].to(self.device)
            masks = batch["masks"].to(self.device)
            targets = batch["targets"].to(self.device)
            sentiment_targets = batch["sentiment_targets"].to(self.device)
            strength_targets = batch["strength_targets"].to(self.device)
            strength_weights = batch["strength_weights"].to(self.device)

            rating, sentiment, strength = self.model(ids, masks)
            loss_rating = self.criterion_rating(rating, targets)
            loss_sentiment = self.criterion_sentiment(sentiment, sentiment_targets)
            loss_strength = self.criterion_strength(strength, strength_targets)
            loss = loss_rating + self.config.loss_weight * (loss_sentiment + loss_strength)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += self.compute_accuracy(rating, targets)[0]

            if step % self.config.print_step == 1:
                print(f"Loss: {loss.item():.4f} | Loss: {total_loss/step:.4f} | Accuracy: {total_acc/step:.4f}")

        self.scheduler.step()
        print(f"End at epoch {epoch} | Loss: {total_loss/step:.4f} | Accuracy: {total_acc/step:.4f}")
        return total_loss / step, total_acc / step

    def valid_epoch(self, epoch):
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_acc_close = 0.0
        total_acc_rating = {i: [0, 0] for i in range(1, 6)}
        total_acc_sentiment = 0.0
        total_acc_strength = 0.0
        total_samples = 0
        # outputs = []
        print("Validation: " + "." * 20)

        for step, batch in enumerate(self.valid_loader, start=1):
            ids = batch["ids"].to(self.device)
            masks = batch["masks"].to(self.device)
            targets = batch["targets"].to(self.device)
            sentiment_targets = batch["sentiment_targets"].to(self.device)
            strength_targets = batch["strength_targets"].to(self.device)
            strength_weights = batch["strength_weights"].to(self.device)

            with torch.no_grad():
                rating, sentiment, strength = self.model(ids, masks)
                loss_rating = self.criterion_rating(rating, targets)
                loss_sentiment = self.criterion_sentiment(sentiment, sentiment_targets)
                loss_strength = self.criterion_strength(strength, strength_targets)
                loss = loss_rating + self.config.loss_weight * (loss_sentiment + loss_strength)

            total_loss += loss.item()
            acc_values = self.compute_accuracy_details(rating, targets)
            total_acc += acc_values[0]
            total_acc_close += acc_values[1]
            for rating_key, acc_by_rating in acc_values[2].items():
                total_acc_rating[rating_key][0] += acc_by_rating[0]
                total_acc_rating[rating_key][1] += acc_by_rating[1]

            acc_sentiment, acc_strength = self.compute_accuracy_other(
                sentiment, sentiment_targets, strength, strength_targets)
            total_acc_sentiment += acc_sentiment
            total_acc_strength += acc_strength
            total_samples += targets.shape[0]

            # _, rating_max_indices = torch.max(rating.data, dim=1)
            # outputs.append(rating_max_indices.detach().cpu().numpy())

        total_loss = total_loss / total_samples
        total_acc = total_acc / total_samples
        total_acc_close = total_acc_close / total_samples
        total_acc_sentiment = total_acc_sentiment / total_samples
        total_acc_strength = total_acc_strength / total_samples
        total_acc_rating = {i: v[0] / max(v[1], 1) for (i, v) in total_acc_rating.items()}
        print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Accuracy: {total_acc:.4f} {total_acc_close:.4f}")
        print(" | ".join([f"rating {i}: {v:.4f}" for (i, v) in total_acc_rating.items()]))
        print(f"Sentiment: {total_acc_sentiment:.4f} | Strength: {total_acc_strength:.4f}")

        # outputs = np.concatenate(outputs, axis=0) + 1
        # print(outputs.shape, outputs[:7], outputs[-7:])
        # np.save(Path(self.pretrained_dir) / self.method_name / "outputs.npy", outputs)
        return total_loss, total_acc

    def run(self):
        for epoch in range(0, self.config.num_epochs):
            print("=" * 30)
            print(f"Epoch {epoch:03d}")
            # self.train_epoch(epoch)
            valid_loss, valid_acc = self.valid_epoch(epoch)

            if valid_acc > self.best_value:
                if self.best_checkpoint is not None:
                    os.remove(self.best_checkpoint)
                self.best_checkpoint = os.path.join(
                    self.checkpoint_dir, f"best_vloss-{valid_loss:.4f}_vacc-{valid_acc:.4f}.pth")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }, self.best_checkpoint)
                self.best_value = valid_acc
            if ((epoch % self.config.save_step == 0) and (epoch >= 90)) or (epoch == self.config.num_epochs - 1):
                checkpoint_name = f"epoch-{epoch:03d}_vloss-{valid_loss:.4f}_vacc-{valid_acc:.4f}"
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }, os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth"))
        return


def main():
    config = load_config(Path.cwd(), "./local_config.yaml")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    print(config)

    model = RobertaClassifier(config)
    runner = Runner(model, config)

    runner.run()
    print(config.method_name)


if __name__ == "__main__":
    main()
