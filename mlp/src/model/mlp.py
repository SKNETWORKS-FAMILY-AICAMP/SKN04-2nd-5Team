import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs.get('input_dim')
        self.hidden_dim = configs.get('hidden_dim')
        self.output_dim = configs.get("output_dim")
        self.dropout_ratio = configs.get('dropout_ratio')
        self.use_batch_norm = configs.get('use_batch_norm')

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.batch_normalization1 = nn.BatchNorm1d(self.hidden_dim)
        self.relu1 = nn.ReLU()  # ReLU
        self.dropout1 = nn.Dropout(p=self.dropout_ratio)

        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.batch_normalization2 = nn.BatchNorm1d(self.hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout_ratio)

        self.output = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.batch_normalization1(x)
        x = self.relu1(x)
        x = self.dropout1(x) # dropout을 적용하여 일부 node 비활성화

        x = self.linear2(x)  # 입력 데이터에 대해 선형 변환을 적용합니다.
        if self.use_batch_norm:
            x = self.batch_normalization2(x)
        x = self.relu2(x)  # ReLU 활성화 함수를 적용하여 비선형성을 추가합니다.
        x = self.dropout2(x) # dropout을 적용하여 일부 node 비활성화

        x = self.output(x)  # 두 번째 선형 변환을 적용하여 최종 출력을 계산합니다.

        return x  # 최종 출력을 반환합니다.
