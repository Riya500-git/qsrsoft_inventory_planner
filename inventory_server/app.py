from flask import Flask, request
import torch
import torch.nn as nn

app = Flask(__name__)

class SVM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1):
        super(SVM, self).__init__()
        self.gnu = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.first = nn.Linear(hidden_size, output_size)
    def forward(self, x): 
        x, _ = self.gnu(x)
        x = self.first(x)
        return x
    
months = {
    0: 0.95,
    1: 0.25,
    2: 0.22,
    3: 0.20,
    4: 0.23,
    5: 0.24,
    6: 0.29,
    7: 0.55,
    8: 0.36,
    9: 0.25,
    10: 0.21,
    11: 0.49
}

@app.route("/inventory")
def hello():
    args = request.args
    location_id = int(args.get("location_id"))
    item_id = int(args.get("item_id"))
    week = int(args.get("week"))
    month = int(args.get("month"))
    k = SVM(4, 1, 100)
    model_state_dict = torch.load('./model.pth')
    k.load_state_dict(model_state_dict)
    x = torch.tensor([[  location_id, item_id, week, months[month]]])
    k.eval()

    # De-normalize the output, as the output is standardized
    mean = 288.3117
    stdev = 417.5643
    prediction = int(k(x).squeeze().item()) * stdev + mean
    return str(prediction)
