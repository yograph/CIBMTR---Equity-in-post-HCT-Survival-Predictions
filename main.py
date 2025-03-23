# ======================
# Imports and Setup
# ======================
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
import optuna

# Define your directories here (adjust as needed)
class Args:
    data_process_dir = './data/'
    encoder_info_dir = './encoder_info/'
    model_dir = './models/'
    submission_dir = './submissions/'
    fold_n = 5

args = Args()

# ======================
# Data Processing Functions
# ======================
def lgb_reg_seconde_process(data_file):
    """
    Process data for LightGBM regressor.
    - Load pickle file.
    - Convert specific columns to categorical.
    - Select input columns and drop unwanted columns.
    - Collect category mappings.
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    # Convert specific columns to category
    for col in data['feature_cat'] + data['feature_onehot']:
        data['X'][col] = data['X'][col].astype('category')
    # Select input columns (combination of features)
    input_columns = data['feature_cat'] + data['feature_value'] + data['feature_onehot']
    data['X'] = data['X'][input_columns]
    # Drop columns not needed
    drop_cols = ['year_hct_trans2cat', 'hla_match_a_high_trans2cat', 
                 'hla_match_b_low_trans2cat', 'comorbidity_score_trans2cat']
    data['X'] = data['X'].drop(drop_cols, axis=1)
    categories = {}
    for col in data['X'].columns:
        if data['X'][col].dtype.name == 'category':
            categories[col] = data['X'][col].cat.categories
    return data, categories

def cat_cls_seconde_process(data_file):
    """
    Process data for CatBoost classifier.
    Similar to the regressor process, but tailored for classification.
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    # Assume similar processing: convert categorical columns
    for col in data['feature_cat'] + data['feature_onehot']:
        data['X'][col] = data['X'][col].astype('category')
    input_columns = data['feature_cat'] + data['feature_value'] + data['feature_onehot']
    data['X'] = data['X'][input_columns]
    # No dropping here (or adjust as needed)
    return data, None

# ======================
# Model Training Functions
# ======================
def lgb_reg_train(seed, fold_n):
    """
    Train a LightGBM regressor using StratifiedKFold.
    - Process data.
    - Create a normalized target.
    - Define sample weights.
    - Perform K-fold training.
    """
    data, categories = lgb_reg_seconde_process(os.path.join(args.data_process_dir, 'data_train.pkl'))
    with open(os.path.join(args.encoder_info_dir, 'categories_lgb_reg.pkl'), 'wb') as f:
        pickle.dump(categories, f)
    
    model_param = {
        'objective': 'regression',
        'min_child_samples': 20,
        'num_iterations': 10000,   # adjust if needed
        'learning_rate': 0.02,
        'extra_trees': True,
        'reg_lambda': 0,
        'reg_alpha': 0,
        'num_leaves': 128,
        'metric': 'mae',
        'max_depth': 6,
        'device': 'cpu',
        'max_bin': 128,
        'verbose': -1,
        'seed': seed,
        'num_threads': 11
    }
    lgb_reg = LGBMRegressor(**model_param)
    
    # Create target: normalized rank within groups defined by efs indicator
    efs_time_norm = data['efs_time'].copy()
    mask1 = data['efs'] == 1
    mask0 = data['efs'] == 0
    efs_time_norm[mask1] = data['efs_time'][mask1].rank() / sum(mask1)
    efs_time_norm[mask0] = data['efs_time'][mask0].rank() / sum(mask0)
    data['efs_time_norm'] = efs_time_norm
    
    # Sample weights
    sample_weight = np.zeros(len(data['efs']))
    sample_weight[mask1] = 0.6
    sample_weight[mask0] = 0.4
    data['sample_weight'] = sample_weight

    sample_num = len(data['X'])
    reg_prediction = np.zeros(sample_num)

    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    y_combine = data['efs'].astype('str') + '|' + data['X']['race_group'].astype('str')
    for i, (train_index, eval_index) in enumerate(skf.split(data['X'], y_combine)):
        print('############fold', i)
        # Prepare training and evaluation data
        efs_train = data['efs'].iloc[train_index]
        efs_eval = data['efs'].iloc[eval_index]
        X_train_reg = data['X'].iloc[train_index, :].copy()
        X_eval_reg = data['X'].iloc[eval_index, :].copy()
        # Add efs as an extra feature
        X_train_reg['efs'] = data['efs'].iloc[train_index].astype('int')
        X_eval_reg['efs'] = data['efs'].iloc[eval_index].astype('int')
        X_train_reg['efs'] = X_train_reg['efs'].astype('category').cat.set_categories([0, 1])
        X_eval_reg['efs'] = X_eval_reg['efs'].astype('category').cat.set_categories([0, 1])
        y_train_reg = data['efs_time_norm'].iloc[train_index]
        y_eval_reg = data['efs_time_norm'].iloc[eval_index]
        
        # Define evaluation set: focus metric on samples where efs==1
        eval_set = [(X_train_reg, y_train_reg), (X_eval_reg[efs_eval==1], y_eval_reg[efs_eval==1])]
        
        # Train model
        lgb_reg.fit(X_train_reg, y_train_reg,
                    eval_set=eval_set[1:2],
                    sample_weight=data['sample_weight'][train_index],
                    callbacks=[
                        # Uncomment if early stopping or logging is needed:
                        # lgb.early_stopping(stopping_rounds=100),
                        # lgb.log_evaluation(period=1000),
                    ])
        
        # OOF prediction
        y_hat_reg = lgb_reg.predict(X_eval_reg)
        reg_prediction[eval_index] = y_hat_reg
        
        # Save model for each fold
        model_info = {'train_index': train_index, 'eval_index': eval_index, 'model': lgb_reg}
        with open(os.path.join(args.model_dir, f'lightgbm_reg_fold{i}.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
    
    # Calculate raw concordance index for efs==1 (assuming you have a concordance_index function defined)
    from lifelines.utils import concordance_index  # or your own implementation
    c_index_overall = concordance_index(data['efs_time'][mask1], reg_prediction[mask1], data['efs'][mask1])
    print('Over All OOF C-index where efs==1:', c_index_overall)
    
def cat_cls_train(fold_n):
    """
    Train a CatBoost classifier with cross-validation.
    (The training code is not fully shown in the notebooks.
     You may want to implement similar logic as in regressor training.)
    """
    # This is a placeholder function.
    # Similar to lgb_reg_train, load data and train CatBoost models across folds.
    # Save each model (with its eval indices) for later use.
    pass

# ======================
# Prediction Functions
# ======================
def cls_predict(fold_n, test):
    """
    Generate predictions for the classification model.
    """
    data_file = os.path.join(args.data_process_dir, 'data_test.pkl' if test else 'data_train.pkl')
    data, _ = cat_cls_seconde_process(data_file)
    with open(os.path.join(args.encoder_info_dir, 'categories_cat_cls.pkl'), 'rb') as f:
        categories = pickle.load(f)
    for col in data['X'].columns:
        if data['X'][col].dtype.name == 'category':
            data['X'][col] = data['X'][col].cat.set_categories(categories[col])
    
    output = np.zeros(len(data['X']))
    for i in range(fold_n):
        with open(os.path.join(args.model_dir, f'catboost_cls_fold{i}.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        model = model_info['model']
        if test:
            y_hat = model.predict_proba(data['X'])[:, 1]
            output += y_hat / fold_n
        else:
            eval_index = model_info['eval_index']
            y_hat = model.predict_proba(data['X'].loc[eval_index])[:, 1]
            output[eval_index] = y_hat
    return output

def reg_predict(fold_n, test):
    """
    Generate predictions for the regression model.
    """
    data_file = os.path.join(args.data_process_dir, 'data_test.pkl' if test else 'data_train.pkl')
    data, _ = lgb_reg_seconde_process(data_file)
    with open(os.path.join(args.encoder_info_dir, 'categories_lgb_reg.pkl'), 'rb') as f:
        categories = pickle.load(f)
    for col in data['X'].columns:
        if data['X'][col].dtype.name == 'category':
            data['X'][col] = data['X'][col].cat.set_categories(categories[col])
    # Force efs to 1 for prediction
    data['X']['efs'] = 1
    data['X']['efs'] = data['X']['efs'].astype('category').cat.set_categories([0, 1])
    
    output = np.zeros(len(data['X']))
    for i in range(fold_n):
        with open(os.path.join(args.model_dir, f'lightgbm_reg_fold{i}.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        model = model_info['model']
        if test:
            y_hat = model.predict(data['X'])
            output += y_hat / fold_n
        else:
            eval_index = model_info['eval_index']
            output[eval_index] = model.predict(data['X'].loc[eval_index])
    return output

# ======================
# Merge and Optimization Functions
# ======================
def merge_fun(Y_HAT_REG, Y_HAT_CLS, a=2.96, b=1, c=0.52):
    """
    Merge function to combine regression and classification predictions.
    a, b, c are parameters to be optimized.
    """
    y_fun = (Y_HAT_REG > 0) * c * np.abs(Y_HAT_REG)**b
    x_fun = (Y_HAT_CLS > 0) * np.abs(Y_HAT_CLS)**a
    res = (1 - y_fun) * x_fun + y_fun
    # Rank the merged predictions
    res = pd.Series(res).rank() / len(res)
    return res

def combine_objective(trial, y_hat_reg, y_hat_cls, efs_time, efs, race_group):
    """
    Objective function for Optuna to optimize merge parameters.
    """
    params = {
        'Y_HAT_REG': y_hat_reg,
        'Y_HAT_CLS': y_hat_cls,
        'a': trial.suggest_uniform("a", 2, 3.5),
        'b': trial.suggest_uniform('b', 0.5, 1.5),
        'c': trial.suggest_uniform('c', 0, 1),
    }
    # Assume you have a scoring function CIBMTR_score returning (score, var_error, metric_list)
    score, var_error, metric_list = CIBMTR_score(efs_time, merge_fun(**params), efs, race_group)
    return score

def merge_param_fit(efs, efs_time, race_group, y_hat_reg, y_hat_cls, fold_n, seed):
    """
    Optimize merge parameters with cross-validation using Optuna.
    """
    y_combine = pd.Series(efs).astype('str') + '|' + pd.Series(race_group).astype('str')
    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    result_combine = np.zeros(len(efs))
    best_params = []
    for i, (train_index, eval_index) in enumerate(skf.split(efs, y_combine)):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: combine_objective(
            trial, 
            y_hat_reg[train_index],
            y_hat_cls[train_index],
            efs_time[train_index],
            efs[train_index],
            race_group[train_index],
        ), n_trials=200)
        result_combine[eval_index] = merge_fun(y_hat_reg[eval_index], y_hat_cls[eval_index], **study.best_params)
        best_params.append(study.best_params)
        print(f'Best merge param fold{i}:', study.best_params)
    return result_combine, best_params

def search_best_merge_params():
    """
    Search for the best merge parameters.
    """
    with open(os.path.join(args.data_process_dir, 'data_train.pkl'), 'rb') as f:
        data = pickle.load(f)
    efs = np.array(data['efs'])
    efs_time = np.array(data['efs_time'])
    race_group = np.array(data['race_group'])
    
    y_hat_cls = cls_predict(fold_n=args.fold_n, test=False)
    auc = roc_auc_score(efs==0, y_hat_cls)
    print('cls auc:', auc)
    
    y_hat_reg = reg_predict(fold_n=args.fold_n, test=False)
    # Using concordance index for samples where efs==1
    from lifelines.utils import concordance_index
    c_index = concordance_index(efs_time[efs==1], y_hat_reg[efs==1])
    print('reg c-index where efs==1:', c_index)
    
    print('search best merge param')
    y_hat_merge, best_params = merge_param_fit(
        efs=efs,
        efs_time=efs_time,
        race_group=race_group,
        y_hat_reg=y_hat_reg,
        y_hat_cls=y_hat_cls,
        fold_n=5,
        seed=888
    )
    
    c_index_overall, var_error, metric_list = CIBMTR_score(efs_time, y_hat_merge, efs, race_group)
    print('OOF Stratified C-index:', c_index_overall)
    
    return best_params

# ======================
# Inference and Submission
# ======================
def generate_submission():
    """
    Generate final merged predictions for test data and create submission CSV.
    """
    # Search for best merge parameters on training data
    merge_param = search_best_merge_params()
    
    # Inference on test data
    y_hat_cls = cls_predict(fold_n=args.fold_n, test=True)
    y_hat_reg = reg_predict(fold_n=args.fold_n, test=True)
    
    # Use the merge function with each set of best parameters from cross-validation and average the result
    y_hat_merge_list = []
    for param in merge_param:
        y_hat_merge_list.append(merge_fun(y_hat_reg, y_hat_cls, **param))
    y_hat_merge = np.mean(np.array(y_hat_merge_list).T, axis=1)
    
    # Load test data IDs for submission
    with open(os.path.join(args.data_process_dir, 'data_test.pkl'), 'rb') as f:
        data_test = pickle.load(f)
    pred = pd.DataFrame({'ID': data_test['ID'], 'prediction': 1 - y_hat_merge})
    pred.to_csv(os.path.join(args.submission_dir, 'submission.csv'), index=False)
    print(pred)

# Uncomment the following lines to run training or generate submission:
# lgb_reg_train(seed=888, fold_n=args.fold_n)
# cat_cls_train(fold_n=args.fold_n)
# generate_submission()
