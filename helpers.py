import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

### Prep

### replace nulls with mode/mean (for categorical/continuous)
def handle_nulls(data):
    for col in data.columns:
        if data[col].isna().sum() >=4890:
            data = data.drop([col], axis=1)
        else:
            if data[col].dtype in ['int64', 'float64']:
                data = data.fillna({col : data[col].mean()})
            else:
                data = data.fillna({col : data[col].mode()[0]})
    
    return data

### security measure for target values bc I keep having issues 
def check_target_vals(target):
    
    red = 0
    
    blue = 0
    
    for val in target:
        if val == 1:
            red += 1
        else:
            blue += 1

    if red != 2859:
        return f"red disconnect - instead of 2859, {red}"
    elif blue != 2037:
        return f"blue disconnect - instead of 2037, {blue}"
    else:
        return "all good"
    
### break into features and target 
def fracture(data):
    target = [1 if victor == 'Red' else 0 for victor in data['Winner']]
    features = data.drop(['Winner'], axis=1)
    return features, target

### categorical and continuous values
def num_and_cat(features):
    num_col = [col for col in features.columns if features[col].dtype in ['int64', 'float64']]
    cat_col = [col for col in features.columns if col not in num_col]
    return num_col, cat_col 

### encode
def encode_if_needed(D):
    enc = LabelEncoder()
    num, cat = num_and_cat(D)
    
    if len(cat) == 0:
        return D
    else:
        for c in cat:
            D[c] = enc.fit_transform(D[c])

### APPROACHES TO FEATURE ENGINEERING 

def dummy_approach(data):
    features, target = fracture(data)
    return features, features.columns, target

def highest_correlating_num_cols(data, squared_thresh):
    features, target = fracture(data)
    desired_cols = []
    
    to_mess_with = features.copy()
    to_mess_with['target'] = target
    
    num_col, _ = num_and_cat(features)
    
    for col in num_col:
        test_val = (to_mess_with[col].corr(to_mess_with['target']))**2

        if test_val >= squared_thresh:
            desired_cols.append(col)
        
    return features, desired_cols, target

def highest_correlating_num_and_cat(data, cat_thresh, squared_thresh, just_cats = False):
    
    features, target = fracture(data)
    
    enc = LabelEncoder()
    desired_cols = []
    
    num_col, cat_col = num_and_cat(features)
    
    for col in cat_col:
        features[col] = enc.fit_transform(features[col])
        
    to_mess_with = features.copy()
    to_mess_with['target'] = target
    
    for col in cat_col:
        test_val = (to_mess_with[col].corr(to_mess_with['target']))**2
        test = test_val >= cat_thresh
        
        if test:
            desired_cols.append(col)
    
    if not just_cats:
        _, nums, unimp = highest_correlating_num_cols(data, squared_thresh)
        for num in nums:
            desired_cols.append(num)
            
    return features, desired_cols, target

def create_diffs(data):
    
    data['draw_diff'] = data['R_draw'] - data['B_draw'] ## draw difference
    data['SigStr_pct_dff'] = data['R_avg_SIG_STR_pct'] - data['B_avg_SIG_STR_pct'] ## mean sig strikes percent 
    data['SigStr_land_diff'] = data['R_avg_SIG_STR_landed'] - data['B_avg_SIG_STR_landed'] ## mean sig strikes 
    data['maj_dec_diff'] = data['R_win_by_Decision_Majority'] - data['B_win_by_Decision_Majority'] ## nean wins 
    data['split_dec_diff'] = data['R_win_by_Decision_Split'] - data['B_win_by_Decision_Split'] ## ^^
    data['unan_dec_diff'] = data['R_win_by_Decision_Unanimous'] - data['B_win_by_Decision_Unanimous'] ## ^^
    data['doc_stop_diff'] = data['R_win_by_TKO_Doctor_Stoppage'] - data['B_win_by_TKO_Doctor_Stoppage'] ##^^
    data['odds_diff'] = data['R_odds'] - data['B_odds'] ## odds
    data['ev_diff'] = data['R_ev'] - data['B_ev'] ## expected value 
    data['subs_diff'] = data['R_avg_SUB_ATT'] - data['B_avg_SUB_ATT'] ## subs attempted
    data['td_landed_diff'] = data['R_avg_TD_landed'] - data['B_avg_TD_landed'] ## takedowns 
    data['td_pct_diff'] = data['R_avg_TD_pct'] - data['B_avg_TD_pct'] ## takedowns percent 
    data['ko_diff'] = data['r_ko_odds'] - data['b_ko_odds'] ## knockouts 
    data['ko_win_diff'] = data['R_win_by_KO/TKO'] - data['B_win_by_KO/TKO'] ## wins by KO/TKO 
    
    return data 

def cols_of_differences(data):
    features, target = fracture(data)
    features = create_diffs(features)
    desired_cols = ['draw_diff', 'SigStr_pct_dff', 'SigStr_land_diff', 'maj_dec_diff', 
          'split_dec_diff', 'unan_dec_diff', 'doc_stop_diff', 'odds_diff', 
          'ev_diff', 'subs_diff', 'td_landed_diff', 'td_pct_diff', 'ko_diff', 
          'ko_win_diff']
    
    return features, desired_cols, target

def differences_and_cat(data, cat_thresh, squared_thresh, just_cats=True):
    
    features, desired_cols, target = cols_of_differences(data)
    _, cats, not_imp = highest_correlating_num_and_cat(data, cat_thresh=cat_thresh, squared_thresh=squared_thresh, just_cats=just_cats)
    for cat in cats:
        desired_cols.append(cat)
    
    return features, desired_cols, target

def discussion_comment_betting_variables(data):
    features, target = fracture(data)
    desired_cols = ['B_current_win_streak', 'R_win_by_Submission', 'B_win_by_Decision_Unanimous', 'R_win_by_Decision_Unanimous', 'R_current_lose_streak', 'B_win_by_TKO_Doctor_Stoppage', 
                    'win_dif', 'B_win_by_Decision_Split', 'B_wins', 'R_Stance', 'B_age', 'B_Weight_lbs', 'R_ev', 'B_total_rounds_fought', 'location', 
                    'R_odds', 'R_Reach_cms', 'R_Weight_lbs', 'R_current_win_streak', 'R_age', 'empty_arena', 'R_win_by_Decision_Split', 'R_draw', 'lose_streak_dif', 'B_draw']
    
    enc = LabelEncoder()
    
    for col in desired_cols:
        if features[col].dtype == 'object':
            features[col] = enc.fit_transform(features[col])
    
    return features, desired_cols, target

def disc_cols_with_differences(data):
    
    features, target = fracture(data)
    
    features = create_diffs(features)
    
    desired_cols = ['B_current_win_streak', 'R_current_lose_streak', 'win_dif', 
                       'R_Stance', 'B_age', 'B_Weight_lbs', 'location', 'R_Reach_cms', 
                       'R_Weight_lbs', 'R_current_win_streak', 'R_age', 'empty_arena', 'lose_streak_dif']
    
    return features, desired_cols, target

### FULL FLOW FOR PREPROCESSING AND DATA PREPARATION 

