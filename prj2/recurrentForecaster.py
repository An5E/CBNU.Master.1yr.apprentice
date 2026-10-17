from torch import nn

class RecurrentForecaster(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size=64, num_layers=2, pred_len=24, dropout=0.1):
        super().__init__()
        if cell_type == "RNN":
            rnn_cls = nn.RNN
        elif cell_type == "LSTM":
            rnn_cls = nn.LSTM
        elif cell_type == "GRU":
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unknown model: {cell_type}")
        self.rnn = rnn_cls(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, pred_len))

    def forward(self, x):
        # x: (B, 96, F)
        output, state = self.rnn(x)
        # 마지막 입력 시점의 최종 계층 표현
        last_hidden = output[:, -1, :] # (B, hidden_size)
        # 24개 미래 시점을 한 번에 예측
        forecast = self.head(last_hidden) # (B, 24)
        return forecast.unsqueeze(-1) # (B, 24, 1)