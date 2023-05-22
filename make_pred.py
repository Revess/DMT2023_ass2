import torch
import pandas as pd
from main import simpleNNRegressor

model = torch.load("./w1_0.ckpt")
model.eval()

df = pd.read_csv("./data/cleaned_test_set_VU_DM.csv").drop(["prop_review_score"],axis=1)
features = torch.tensor(((df - df.min()) / (df.max() - df.min())).drop(['srch_id', 'prop_id'], axis=1).to_numpy()).to("cuda:0").to(torch.float32)
df = df[['srch_id', 'prop_id']]
preds = model(features)
preds = preds.cpu().detach().numpy()
df['rating'] = preds
df.sort_values('rating', ascending=False).drop(['rating'],axis=1).to_csv("./pred.csv",index=False)
print(df.shape)