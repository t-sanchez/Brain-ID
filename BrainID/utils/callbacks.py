import os
from lightning.pytorch.callbacks import Callback
import pandas as pd

class LogPredictionsCallback(Callback):
    def __init__(self, output_dir:str, output_file: str = "predictions.csv"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_file = os.path.join(output_dir, output_file)
        self.predictions = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        label = batch["label"]
        pred = outputs["preds"][0]["pred"]
        # softmax
        prob_good = label[:, 1]
        pred_good = pred.softmax(dim=-1)[:, 1]
        
        # Optional: convert to CPU + numpy
        prob_good = prob_good.detach().cpu().numpy()
        pred_good = pred_good.detach().cpu().numpy()
        path = batch["path"]
        name = path.split("/")[-1].split("_")[0]
        # Store predictions for later saving
        self.predictions.extend(zip(name, pred_good, prob_good, path))

    def on_test_end(self, trainer, pl_module):
        # Convert predictions to DataFrame
        df = pd.DataFrame(
            self.predictions, 
            columns=["name", "pred_good", "prob_good", "path"]
        )
        # Save to CSV
        df.to_csv(self.output_file, index=False)
        print(f"Predictions saved to {self.output_file}")