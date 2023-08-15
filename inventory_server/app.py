from flask import Flask, request
from flask_cors import CORS, cross_origin
import torch
import torch.nn as nn
from math import sqrt
import datetime

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Model class, copied over from the notebook file
class SVM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1):
        super(SVM, self).__init__()
        self.gnu = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.first = nn.Linear(hidden_size, output_size)
    def forward(self, x): 
        x, _ = self.gnu(x)
        x = self.first(x)
        return x

# Mapping of month to customer traffic
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

# Mapping of item_id to base_price
item_list = {
    1: 2,
    2: 4,
    3: 10,
    4: 20,
    5: 40,
    6: 60
}

with open("locations.txt") as f:
    lines = f.readlines()
    locations = [eval(line) for line in lines]

k = SVM(4, 1, 100)
model_state_dict = torch.load('./model.pth')
k.load_state_dict(model_state_dict)

curr_time = datetime.date(2023, 8, 15) #curr_time maps to week 520

@app.route("/inventory")
@cross_origin()
def hello():
    args = request.args
    lat = int(args.get("lat"))
    lon = int(args.get("lon"))

    location_finder = [sqrt(pow(stored_lat - lat, 2) + pow(stored_lon - lon, 2)) \
                        for stored_lat, stored_lon in locations]
    location_id = location_finder.index(min(location_finder))
    item_id = int(args.get("item_id"))
    date = datetime.datetime.strptime(args.get("date"), '%Y-%m-%d').date()
    delta = curr_time - date
    week_num = 520 - delta.days / 7

    x = torch.tensor([[  location_id, item_id, week_num, date.month]])
    k.eval()

    # De-normalize the output, as the output is standardized
    mean = 288.3117
    stdev = 417.5643
    prediction = int(k(x).squeeze().item()) * stdev + mean
    return str(prediction)
