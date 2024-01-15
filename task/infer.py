from typing import List, Dict, Optional,Text
from data_utils.load_data import Get_Loader
import torch
import os
import pandas as pd
from tqdm import tqdm
from eval_metric.evaluate import ScoreCalculator
from model.build_model import build_model

class Predict:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.model = build_model(config)
        self.dataloader = Get_Loader(config)
    def predict_submission(self):
        # Load the model
        print("Loading the best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Obtain the prediction from the model
        print("Obtaining predictions...")
        test =self.dataloader.load_test()
        submits=[]
        ids=[]
        self.model.eval()
        with torch.no_grad():
            for it, (sents, id) in enumerate(tqdm(test)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits = self.model(sents)
                submits.extend(logits.cpu().numpy())
                if isinstance(id, torch.Tensor):
                    ids.extend(id.tolist())
                else:
                    ids.extend(id)
                    
        data = {'id': ids,'generated': submits}
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)