import turicreate as tc
import pandas as pd

model = tc.load_model("./ffm_cktp/")

test_data = pd.read_csv("./data/testing_ffm.csv").drop(['prop_review_score'],axis=1)
pred_data = tc.SFrame(test_data)
print(pred_data.shape)
preds = model.recommend(pred_data['srch_id'], new_observation_data=pred_data)
df = pd.read_csv("./data/cleaned_test_set_VU_DM.csv").drop(["prop_review_score"],axis=1)
df['rating'] = preds
df.sort_values('rating', ascending=False).drop(['rating'],axis=1).to_csv("./ffm_pred.csv",index=False)
print(df.shape)

# 4959183
