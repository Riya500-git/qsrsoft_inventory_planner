# QSRSoft Inventory Planner #

Packages that contain all the logic for the inventory planner.

When used together, these packages aim to provide a prediction for the amount of inventory needed for a restaurant given a location and a date (can be both past and future).

## How it works ##

`inventory_server` contains the server logic. It can be started locally with the following commands:
```
cd inventory_server
pip install -r requirements.txt
python3 -m flask run
```

`website` contains the website logic. It will also be hosted on a public endpoint that accesses localhost, so it will require the above server to be deployed locally to use successfully. It can also be run locally with the following commands:
```
cd website
npm install
npm start
```

`inventory.ipynb` contains the Notebook logic that was used to train the ML model that aims to predict the inventory amount.

