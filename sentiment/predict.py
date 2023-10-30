from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import DataLoader

from sentiment.dataset import ReviewsDataset
from sentiment.model import RobertaClassifier


def main():
    config = OmegaConf.load("./local_config.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    if config.embedding_model == "bert":
        tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    else:
        tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    data_dir = config.data_dir
    test_dataset = ReviewsDataset(Path(data_dir) / "review_test_data.csv", tokenizer, config)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers)

    # Model
    model = RobertaClassifier(config)
    model.eval()
    model.to(device)

    pretrained_dir = "../weights"
    method_name = "roberta_fc-False_pooler-False_transformer-True_d0.0_wd1e-4_lr1e-4"
    model.load_state_dict(
        torch.load(Path(pretrained_dir) / method_name / "best_vloss-0.5662_vacc-0.9325.pth")["model"]
    )

    # Run
    outputs = []

    for step, batch in enumerate(test_loader, start=1):
        ids = batch["ids"].to(device)
        masks = batch["masks"].to(device)

        with torch.no_grad():
            rating, sentiment, strength = model(ids, masks)
            _, rating_max_indices = torch.max(rating.data, dim=1)
            outputs.append(rating_max_indices.detach().cpu().numpy())
        break

    outputs = np.concatenate(outputs, axis=0) + 1
    print(outputs.shape)
    np.save(Path(pretrained_dir) / method_name / "outputs____.npy", outputs)


if __name__ == "__main__":
    main()
