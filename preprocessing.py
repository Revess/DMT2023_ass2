import pandas as pd
import numpy as np
import fire, os

def calc_CTR(click, book, TC, w_1=0, w_2=1): #(click, book, TC, w_1=0.1, w_2=2)
    return min(0 if click == 0 and book == 0 else w_2/TC if click > 0 and book == 1 else w_1*1/TC ,1)

def season_booking(row):
    month = row['date_booking'].month
    day = row['date_booking'].day
    if (month == 3 and day >= 21) or (month == 4) or (month == 5) or (month == 6 and day < 21):
        return 0#'spring'
    elif (month == 6 and day >= 21) or (month == 7) or (month == 8) or (month == 9 and day < 21):
        return 1#'summer'
    elif (month == 9 and day >= 21) or (month == 10) or (month == 11) or (month == 12 and day < 21):
        return 2#'autumn'
    else:
        return 3#'winter'

def process(path):
    df = pd.read_csv(path)
    if "train" in path:
        df = df.drop(['position', 'gross_bookings_usd'], axis=1)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['weekday'] = df['date_time'].dt.dayofweek
    df['part_of_day'] = pd.cut(df['date_time'].dt.hour,[0,6,12,18,24],labels=[0,1,2,3],include_lowest=True) # 'night','morning','afternoon','evening' , 0-4
    df['last_minute'] = [1 if x <= 14 else 0 for x in df['srch_booking_window']]
    df['date_booking'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'], unit='d')
    # if "train" in path:
    #     df = df[(df['price_usd'] >= df['price_usd'].quantile(0.001)) & (df['price_usd'] <= df['price_usd'].quantile(0.999))]
    df['date_booking'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'], unit='d')
    df['season_booking'] = df.apply(season_booking, axis=1)
    df.drop(['date_booking'], axis=1, inplace=True)
    #If the price is outside the lower and upper quantile, replace it with the mean of the price of that hotel
    df['price_usd'] = np.where((df['price_usd'] < df['price_usd'].quantile(0.001)) | (df['price_usd'] > df['price_usd'].quantile(0.999)), df.groupby('prop_country_id')['price_usd'].transform('mean'), df['price_usd'])
    # df['prop_review_score'] = df['prop_review_score'].fillna(method='bfill')
    df['comp_rate'] = [1 if all(x == 1 for x in row) else 0 if all(x == 0 for x in row) else -1 for row in df[['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']].values]
    columns = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']
    for column in columns:
        df[column] = df[column] + 1
        df['comp_rate'] = df.groupby('prop_id')[column].transform('mean')
    columns2 = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']
    for column in columns2:
        df[column] = df[column] + 1
        df['comp_inv'] = df.groupby('prop_id')[column].transform('mean')
    columns2 = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
    for column in columns2:
        df['comp_rate_percent_diff'] = df.groupby('prop_id')[column].transform('mean')
    df = df.dropna(axis=1, how="any")
    if "train" in path:
        df['rating'] = pd.merge(
        df[['srch_id','click_bool', 'booking_bool']].groupby('srch_id').sum().reset_index().drop("booking_bool", axis=1).rename(columns={'click_bool':"summed_click_bool"}), 
        df[['srch_id','click_bool', 'booking_bool']], 
        how='left', 
        on= 'srch_id').apply(lambda row: calc_CTR(row['click_bool'], row['booking_bool'], row['summed_click_bool']) ,axis=1)
        df = df.drop(['booking_bool', 'click_bool'],axis=1)
    df = df.drop(['date_time', 'site_id'],axis=1)
    if "train" in path:
        df = df.dropna(axis=0)
    df.to_csv(f"./data/cleaned_{os.path.basename(path)}", index=False)


if __name__ == "__main__":
    fire.Fire(process)