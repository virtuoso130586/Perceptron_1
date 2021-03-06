import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data
import logging
import os
from utils import all_utils
import joblib
import numpy as np

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str)



AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(AND)
X,y = prepare_data(df)
logging.info(f"x-{X} ,y-{y}")
ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()
logging.info(f"saving or.model")
all_utils.save_model(model, "or.model")
inputs = np.array([[1,1], [0,1]])
loaded_model = joblib.load("models/or.model")
loaded_model.predict(inputs)