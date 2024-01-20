import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA 

import matplotlib.pyplot as plt

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

### full flow to create all different approach datasets (NOT ENCODED)
def data_prep_and_feat_engineering(data, cat_thresh, squared_thresh):
    
    ### handle nulls
    data = handle_nulls(data)
    
    ### dictionary of approaches 
    approach_dict = {1 : dummy_approach,
                     2 : highest_correlating_num_cols,
                     3 : highest_correlating_num_and_cat,
                     4 : cols_of_differences,
                     5 : differences_and_cat,
                     6 : discussion_comment_betting_variables,
                     7 : disc_cols_with_differences}
    
    results_dict = {}
    
    for i in range(1,8):
        
        if i == 2:
            features, desired_cols, target = approach_dict[i](data, squared_thresh)
        elif i in [3, 5]:
            features, desired_cols, target = approach_dict[i](data, cat_thresh, squared_thresh)
        else:
            features, desired_cols, target = approach_dict[i](data)
        
        approach_name = f"approach {i}"
        
        results_dict[approach_name] = (features, desired_cols, target)
    
    return results_dict

### encoding and breaking down each individual approach bundle 
def break_down_bundle(bundle):
    
    enc = LabelEncoder()
    
    D = bundle[0]
    f = bundle[1]
    y = bundle[2]
    X = D[f]
    
    _, cats = num_and_cat(X)
    if len(cats) > 0:
        for cat in cats:
            X[cat] = enc.fit_transform(X[cat])
    
    return X, y

def majority_vote(model_list, features, target):
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, test_size=0.2)
    
    preds_dict = {}
    
    i = 1
    
    final_preds = []
    avg_vals = []
    
    for model in model_list:
        model.fit(X_train, y_train)
        model_title = f"model {i}"
        model_preds = model.predict(X_test)
        preds_dict[model_title] = model_preds
        ind_val = accuracy_score(model_preds, y_test)
        print(f"individual score for model {i} : {ind_val}")
        avg_vals.append(ind_val)
        i += 1
    
    for j in range(980):
        
        decision = 0
        
        for key in preds_dict.keys():
            
            decision += preds_dict[key][j]
            
        if decision < 3:
            
            final_preds.append(0)
        
        else:
            
            final_preds.append(1)
    
    final_preds_array = np.array(final_preds)
    
    print("")
    print(f"Accuracy score for majority vote: {accuracy_score(final_preds_array, y_test)}")
    print(f"Average of individual accuracy scores: {sum(avg_vals)/len(avg_vals)}")
    print("******************************************************************************")
    print("")
    
def pca_graph(data):
    scale = StandardScaler()
    pca = PCA()
    
    data_stand = scale.fit_transform(data)
    data_pca = pca.fit_transform(data_stand)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()
    
    return data_pca

def pca_execute(data, data_pca, optimal_num):
    selected_components = data_pca[:, :optimal_num]
    components_df = pd.DataFrame(data=selected_components, columns=[f'PC{i+1}' for i in range(optimal_num)])
    results_df = pd.concat([data, components_df], axis=1)
    
    return results_df, components_df