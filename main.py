# ライブラリのインポート
import re
import random
import time
from statistics import mode
from tqdm import tqdm  # 進捗バー表示のためのライブラリ

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, T5Tokenizer
from torch.nn.utils.rnn import pad_sequence


# シード値の決定（再現性の担保）
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# テキストの前処理
def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # 冠詞の削除
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't",
        "isnt": "isn't",
        "arent": "aren't",
        "wont": "won't",
        "cant": "can't",
        "wouldnt": "wouldn't",
        "couldnt": "couldn't",
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", " ", text)

    # 句読点をスペースに変換
    text = re.sub(r"\s+,", ",", text)

    # 連続するスペースを1つに変換
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 1. データローダーの作成
#
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        # bertトークナイザーのロード
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # 質問文に前処理を適用
        self.df["question"] = self.df["question"].apply(process_text)

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : dict
            質問文をトークン化したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}").convert("RGB")  # RGB画像に変換
        image = self.transform(image)
        question = self.df["question"][idx]
        question_inputs = self.tokenizer(
            question, return_tensors="pt", padding="max_length", max_length=50, truncation=True
        )

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, question_inputs, torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, question_inputs

    def __len__(self):
        return len(self.df)


# カスタムコラート関数
def collate_fn(batch):
    images, question_inputs, answers, mode_answer_idx = zip(*batch)
    images = torch.stack(images)
    input_ids = pad_sequence([qi["input_ids"].squeeze(0) for qi in question_inputs], batch_first=True)
    attention_mask = pad_sequence([qi["attention_mask"].squeeze(0) for qi in question_inputs], batch_first=True)
    answers = pad_sequence(answers, batch_first=True)
    mode_answer_idx = torch.tensor(mode_answer_idx)
    return images, {"input_ids": input_ids, "attention_mask": attention_mask}, answers, mode_answer_idx


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.0

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.0
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


# 3. モデルの実装
class VQAModel(nn.Module):
    def __init__(self, vocab_size, n_answer):
        super().__init__()
        self.resnet = ResNet18()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.fc = nn.Sequential(
            nn.Linear(768 + 512, 512), nn.ReLU(inplace=True), nn.Linear(512, n_answer)  # 回答の数に合わせて出力層を調整
        )

    def forward(self, image, question_inputs):
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.bert(**question_inputs).last_hidden_state[
            :, 0, :
        ]  # BERTの[CLS]トークンの出力を特徴量として使用

        x = torch.cat([image_feature, question_feature], dim=1)
        output = self.fc(x)

        return output


# 回答生成関数の追加
def generate_answer(latent_representation, answer_generator, tokenizer, device):
    latent_str = " ".join(map(str, latent_representation.cpu().detach().numpy().flatten().tolist()))
    input_ids = tokenizer("generate answer: " + latent_str[:1000], return_tensors="pt").input_ids  # 長さ制限を追加
    input_ids = input_ids.to(device)  # デバイスを統一
    outputs = answer_generator.generate(input_ids, max_new_tokens=50)  # max_new_tokensを設定
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    with tqdm(total=len(dataloader), desc="Training") as pbar:  # 進捗バーを追加
        for image, question, answers, mode_answer in dataloader:
            image, question = image.to(device), {k: v.to(device) for k, v in question.items()}
            answers, mode_answer = answers.to(device), mode_answer.to(device)

            output = model(image, question)
            loss = criterion(output, mode_answer.squeeze().long())  # 出力に対して損失を計算

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += VQA_criterion(output.argmax(1), answers)  # VQA accuracy
            simple_acc += (output.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

            pbar.update(1)  # 進捗バーを更新

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


# モデルの評価
def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    with tqdm(total=len(dataloader), desc="Evaluating") as pbar:  # 進捗バーを追加
        for image, question, answers, mode_answer in dataloader:
            image, question = image.to(device), {k: v.to(device) for k, v in question.items()}
            answers, mode_answer = answers.to(device), mode_answer.to(device)

            output = model(image, question)
            loss = criterion(output, mode_answer.squeeze().long())  # 出力に対して損失を計算

            total_loss += loss.item()
            total_acc += VQA_criterion(output.argmax(1), answers)  # VQA accuracy
            simple_acc += (output.argmax(1) == mode_answer).mean().item()  # simple accuracy

            pbar.update(1)  # 進捗バーを更新

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # 水平反転の追加
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色調補正の追加
            transforms.ToTensor(),
        ]
    )

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = VQAModel(vocab_size=len(train_dataset.question2idx) + 1, n_answer=len(train_dataset.answer2idx)).to(
        device
    )  # VQAModelの初期化時に引数を渡す

    # 回答生成用のモデルとトークナイザをロード
    answer_generator = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # optimizer / criterion
    num_epoch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # 初期学習率を0.001→0.01へ変更
    # 学習率スケジューラの追加
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(
            f"【{epoch + 1}/{num_epoch}】\n"
            f"train time: {train_time:.2f} [s]\n"
            f"train loss: {train_loss:.4f}\n"
            f"train acc: {train_acc:.4f}\n"
            f"train simple acc: {train_simple_acc:.4f}"
        )
        scheduler.step()  # スケジューラをステップ

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question_inputs in test_loader:
        image, question_inputs = image.to(device), {k: v.to(device) for k, v in question_inputs.items()}
        latent_representation = model(image, question_inputs)
        pred = generate_answer(latent_representation, answer_generator, tokenizer, device)
        submission.append(pred)

    # 提出フォーマットの調整
    # submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)


if __name__ == "__main__":
    main()
