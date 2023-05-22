import turicreate as tc
import pandas as pd

df = pd.read_csv("./data/cleaned_training_set_VU_DM.csv").drop(['prop_review_score'],axis=1)
actions = tc.SFrame(df)
training_data, validation_data = tc.recommender.util.random_split_by_user(actions, 'srch_id', "prop_id")
model = tc.recommender.create(training_data, 'srch_id', "prop_id", target='rating')
model.summary()
model.evaluate(validation_data)
model.save("./ffm_cktp")