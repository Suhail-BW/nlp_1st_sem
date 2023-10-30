import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel


class FC_BN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        x = self.fc(inputs)
        out = self.bn(x)
        return out


class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=-1):
        super(Conv_BN, self).__init__()
        if padding == -1:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        x = self.conv(inputs)
        out = self.bn(x)
        return out


class LSTM_BN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(LSTM_BN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=out_channels,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.bn = nn.LayerNorm(out_channels * 2)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        x = self.lstm(inputs)[0]
        out = self.bn(x)
        out = out.permute(0, 2, 1)
        return out


class RobertaClassifier(nn.Module):
    def __init__(self, config):
        super(RobertaClassifier, self).__init__()
        self.use_pooler = config.use_pooler
        self.fc_only = config.fc_only
        self.use_lexicon = config.use_lexicon
        self.use_transformer = config.use_transformer

        assert config.embedding_model in ["bert", "roberta"]
        if config.embedding_model == "bert":
            self.embedding = BertModel.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        elif config.embedding_model == "roberta":
            self.embedding = RobertaModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.embedding.pooler.parameters():
            param.requires_grad = self.use_pooler

        print(self.embedding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.conv1_1 = Conv_BN(in_channels=768, out_channels=768, kernel_size=1)
        # self.conv2_3 = Conv_BN(in_channels=768, out_channels=768, kernel_size=3)
        # self.conv2_5 = Conv_BN(in_channels=768, out_channels=768, kernel_size=5)
        # self.conv2_7 = Conv_BN(in_channels=768, out_channels=768, kernel_size=7)
        self.lstm1 = LSTM_BN(in_channels=768, out_channels=256, dropout=config.dropout)
        self.conv1 = Conv_BN(in_channels=512, out_channels=256, kernel_size=5)
        self.lstm2 = LSTM_BN(in_channels=256, out_channels=128, dropout=config.dropout)
        self.conv2 = Conv_BN(in_channels=256, out_channels=128, kernel_size=5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc1 = FC_BN(768, 768)
        self.fc2 = FC_BN(768, 128)
        self.fc3 = nn.Linear(128, 128)

        in_channels = 768 if self.use_pooler else 128
        self.fc_out_rating = nn.Linear(in_channels, config.num_classes)
        self.fc_out_sentiment = nn.Linear(in_channels, 3)
        self.fc_out_strength = nn.Linear(in_channels, 1)

    def forward(self, input_ids, attention_mask):
        roberta_outputs = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        per_token_output = roberta_outputs[0].permute(0, 2, 1)
        pooler_output = roberta_outputs[1]

        # FC branch
        if self.fc_only and self.use_pooler:
            pooler = pooler_output
            rating = self.fc_out_rating(pooler)
            sentiment = self.fc_out_sentiment(pooler)
            strength = self.fc_out_strength(pooler).squeeze()
            return rating, sentiment, strength
        elif self.fc_only:
            pooler = per_token_output[..., 0]
            pooler = self.dropout(self.relu(self.fc1(pooler)))
            pooler = self.dropout(self.relu(self.fc2(pooler)))
            pooler = self.relu(self.fc3(pooler))

            rating = self.fc_out_rating(pooler)
            sentiment = self.fc_out_sentiment(pooler)
            strength = self.fc_out_strength(pooler).squeeze()
            return rating, sentiment, strength

        # CNN branch
        # x3 = self.max_pool(self.relu(self.conv2_3(x))).squeeze()
        # x5 = self.max_pool(self.relu(self.conv2_5(x))).squeeze()
        # x7 = self.max_pool(self.relu(self.conv2_7(x))).squeeze()
        # x = x3 + x5 + x7
        x = per_token_output
        if not self.use_transformer:
            x = self.lstm1(x)
            x = self.relu(self.conv1(x))
            x = self.lstm2(x)
            x = self.relu(self.conv2(x))
            pooler = self.max_pool(x).squeeze()

        else:
            x = x.permute(0, 2, 1)
            key_padding_mask = (attention_mask == 0).bool()
            x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
            pooler = x[:, 0, :]
            pooler = self.dropout(self.relu(self.fc2(pooler)))

        pooler = self.dropout(self.relu(self.fc3(pooler)))
        rating = self.fc_out_rating(pooler)
        sentiment = self.fc_out_sentiment(pooler)
        strength = self.fc_out_strength(pooler).squeeze()

        return rating, sentiment, strength
