import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from skimage.transform import resize
from torchvision import transforms, models


FIXED_TEST_SET = True  # True: Test set을 고정하여 사용
TEST_SAMPLES_PER_CLASS = 100  # 각 클래스에서 50개씩 선택

class ScalogramDataset(Dataset):
    def __init__(self, root_dir, imbalance_factors, transform=None, simulate_imbalance=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data_samples = []  # 전체 샘플 리스트
        self.class_map = {
            "05cNm": 0, "10cNm": 1, "20cNm": 2, "30cNm": 3,
            "40cNm": 4, "50cNm": 5, "60cNm": 6
        }
        self.simulate_imbalance = simulate_imbalance
        self.imbalance_factors = imbalance_factors
        self._load_data()

    def _load_data(self):
        all_samples = defaultdict(list)  # {class_label: [(sample, label), ...]}
        np.random.seed(0)  # 랜덤 시드 고정 (재현성 확보)
        random.seed(0)

        # 각 클래스 폴더에서 .npy 파일 읽기
        for class_folder in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            # class_map에 등록된 폴더만 처리
            if class_folder not in self.class_map:
                continue

            class_label = self.class_map[class_folder]

            for file_name in tqdm(os.listdir(class_path), desc=f"Loading {class_folder}"):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(class_path, file_name)
                    scalogram_data = np.load(file_path)  # shape: (samples, freq, time)

                    for i in range(scalogram_data.shape[0]):
                        sample = scalogram_data[i, :, :]
                        sample = (sample - sample.min()) / (sample.max() - sample.min()) * 255
                        sample = Image.fromarray(sample.astype(np.uint8)).convert('RGB')

                        if self.transform:
                            sample = self.transform(sample)

                        all_samples[class_label].append((sample, class_label))

        # Step 1: Test 데이터 고정 (클래스별 50개씩 선택)
        print("⚠️ 새로운 Test set을 생성합니다.")
        test_indices = {}
        test_samples = []
        for class_label, samples in all_samples.items():
            test_indices[class_label] = random.sample(range(len(samples)), min(TEST_SAMPLES_PER_CLASS, len(samples)))

        # Step 2: Test 데이터 분리
        trainval_samples = []
        for class_label, samples in all_samples.items():
            selected_test_indices = test_indices[class_label]
            test_samples.extend([samples[i] for i in selected_test_indices])
            trainval_samples.extend([samples[i] for i in range(len(samples)) if i not in selected_test_indices])

        # reverse mapping: {label: class_folder 이름}
        reverse_class_map = {v: k for k, v in self.class_map.items()}

        # Step 3: Imbalance 적용 (Train/Val)
        if self.simulate_imbalance:
            # 먼저 클래스별로 그룹화
            grouped_trainval = defaultdict(list)
            for sample, label in trainval_samples:
                grouped_trainval[label].append((sample, label))

            filtered_samples = []
            for label, samples in grouped_trainval.items():
                class_name = reverse_class_map[label]
                factor = self.imbalance_factors.get(class_name, 1.0)
                allowed_count = int(len(samples) * factor)
                # reproducibility를 위해 셔플 후 선택
                random.shuffle(samples)
                filtered_samples.extend(samples[:allowed_count])
        else:
            filtered_samples = trainval_samples

        self.data_samples = filtered_samples  # Train/Val 데이터
        self.test_samples = test_samples   # Test 데이터 저장

        # 전체 샘플 수 출력
        total_all_samples = sum(len(v) for v in all_samples.values())
        print(f"전체 샘플: {total_all_samples}, Train/Val: {len(self.data_samples)}, Test: {len(test_samples)}")

        # 클래스별 전체 샘플 수
        print("클래스별 전체 샘플 수:")
        for class_label, samples in all_samples.items():
            print(f"  {reverse_class_map[class_label]}: {len(samples)}개")

        # 클래스별 Train/Val 샘플 수 (Imbalance 적용 후)
        class_trainval_counts = defaultdict(int)
        for _, label in self.data_samples:
            class_trainval_counts[label] += 1
        print("클래스별 Train/Val 샘플 수 (Imbalance 적용 후):")
        for class_label, count in class_trainval_counts.items():
            print(f"  {reverse_class_map[class_label]}: {count}개")

        # 클래스별 Test 샘플 수
        class_test_counts = defaultdict(int)
        for _, label in test_samples:
            class_test_counts[label] += 1
        print("클래스별 Test 샘플 수:")
        for class_label, count in class_test_counts.items():
            print(f"  {reverse_class_map[class_label]}: {count}개")

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample, label = self.data_samples[idx]
        label = torch.tensor(label, dtype=torch.long)
        return sample, label

def create_dataset(root_dir):
    transform = transforms.ToTensor()
    dataset = ScalogramDataset(
        root_dir=root_dir,
        transform=transform,  # PIL Image를 tensor로 변환
        simulate_imbalance=True
    )

    # Train/Val/Test 나누기
    total_samples = len(dataset.data_samples)  # Train+Val 샘플 개수
    train_size = int(0.8 * total_samples)

    trainval_indices = list(range(total_samples))
    random.shuffle(trainval_indices)

    train_indices = trainval_indices[:train_size]
    val_indices = trainval_indices[train_size:]

    # PyTorch Subset 적용
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = dataset.test_samples  # Test는 따로 저장됨

    print(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")

    return dataset, train_dataset, val_dataset, test_dataset

def iterate_train(iteration, train_loader, val_loader, test_loader):
    accs = []
    f1s = []
    for i in range(iteration):
        # 루프에서 torch.random_seed() 지정하지 않고 학습
        model = models.googlenet(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # ----------------------------------
        # 7. 모델 학습 루프
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            train_accuracy = correct_train / total_train
            train_loss = running_loss / len(train_loader)

            model.eval()
            correct_val = 0
            total_val = 0
            running_val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

            val_accuracy = correct_val / total_val
            val_loss = running_val_loss / len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "../ckpt/best_model.pth")

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # ----------------------------------
        # 8. 테스트 평가 : 전체 Accuracy와 Macro F1 Score, Classification Report, Confusion Matrix 계산
        model.load_state_dict(torch.load("../ckpt/best_model.pth"))
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        accs.append(overall_acc)
        f1s.append(macro_f1)

    accs = np.array(accs)
    f1s = np.array(f1s)

    print(accs, np.mean(accs), np.std(accs))
    print(f1s, np.mean(f1s), np.std(f1s))

if __name__ == "__main__":
    transform = transforms.ToTensor()

    root_dir = "/data/resized/measurementSeries_B"
    imbalance_factors = {
        "05cNm": 0.005,
        "10cNm": 0.005,
        "20cNm": 1.0,
        "30cNm": 1.0,
        "40cNm": 1.0,
        "50cNm": 1.0,
        "60cNm": 1.0
    }
    dataset, train_dataset, val_dataset, test_dataset = create_dataset(root_dir, imbalance_factors)

    print(f"Total samples: {len(dataset)}")
    print(f"Class mapping: {dataset.class_map}")

    # Example data sample 확인
    sample, label = dataset[0]

    # 방법 1 : PIL Image의 size 속성 사용 (width, height)
    print(f"Sample size (width, height): {sample.size()}, Label: {label}")

    # 방법 2 : NumPy 배열로 변환하여 shape 확인 (height, width, channels)
    sample_np = np.array(sample)
    print(f"Sample shape: {sample_np.shape}, Label: {label}")


    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    lr = 1e-3
    num_epochs = 10

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    iterate_train(5)