import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data
import logging
import os
from utils.all_utils import prepare_data, save_model, save_plot
import joblib # FOR SAVING MY MODEL AS A BINARY FILE
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight") # THIS IS STYLE OF GRAPHS

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str)


logging.info(f"preparing the training data")
AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)
X,y = prepare_data(df)
logging.info(f"x-{X} ,y-{y}")
ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()

save_model(model, "and.model")
inputs = np.array([[1,1], [0,1]])
loaded_model = joblib.load("models/and.model")
loaded_model.predict(inputs)
save_plot(df, "and.png", model)