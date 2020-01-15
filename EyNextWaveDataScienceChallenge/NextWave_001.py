import numpy as np # linear algebra
import pandas as pd # data processing
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

#seed used on random generators
seed =123


#evaluation function
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) 
    return 'f1', f1_score(y_true, y_hat), True

#parameters from Light GBM with Dart mode 
def train_lgb(defseed, d_train, d_valid):
    params = {   
		'boosting_type': 'dart',
		'objective': 'xentropy',
		'learning_rate': 0.07,
		'max_depth': 5,
		'subsample': 0.6,
        'feature_fraction': 0.7,
		'colsample_bytree': 0.7,
		'alpha': 0.5,
		'random_state': defseed,
        'verbosity' :-1,
        'num_threads': -1,
        'xgboost_dart_mode': True,
        'uniform_drop': True,
        'lambda_l1': 0.4,
        'lambda_l2': 0.7,
        'bagging_seed': defseed,
        'bagging_fraction': 0.7,
        'bagging_freq': 3,
        'drop_rate': 0.3,
        'drop_seed':defseed,
        'cat_smooth': 18.0,
        
       }    
    
    model = lgb.train(params, 
            train_set=d_train, 
            valid_sets=[d_train, d_valid], 
            feval=lgb_f1_score,
            num_boost_round=800,
            early_stopping_rounds=100,
            verbose_eval=50)
        
    return model



#function to tranform the time feature in the base 
def split_datetime(df, value = 0):
    if value == 0:
        split = pd.DataFrame(df.time_entry.str.split(':',2).tolist(), columns = ['hour','minute','second'])
        string = 'time_entry'
    else:
        split = pd.DataFrame(df.time_exit.str.split(':',2).tolist(), columns = ['hour','minute','second'])
        string = 'time_exit'
    df[string] = split['second'].apply(int) + (split['minute'].apply(int) * 60) + (split['hour'].apply(int) * 3600)
    return df


#Loading the base
train = pd.read_csv("G:\\Meu Drive\\LP&D\\NextWave\\data_train.csv", delimiter=',')
test = pd.read_csv("G:\\Meu Drive\\LP&D\\NextWave\\data_test.csv", delimiter=',')

# x and y from the city center 
lim_esq = 3750901.5068  
lim_dir = 3770901.5068
lim_inf = -19268905.6133
lim_sup = -19208905.6133

#middle point of the city center
x_centro = lim_esq+lim_dir/2
y_centro = lim_inf+lim_sup/2

#dropping the columns that we don't use 
train = train.drop(['Unnamed: 0','vmax', 'vmin', 'vmean'], axis = 1)
test = test.drop(['Unnamed: 0','vmax', 'vmin', 'vmean'], axis = 1)


#feature creation 

#%% translating the time in the base to TimeStamp fomr
# this feature don't change almost nothing in the score  
train['time_entry_seconds'] = [sum(x * int(t) for x, t in zip([3600, 60, 1], times.split(":"))) for times in train['time_entry']] 
train['time_exit_seconds'] = [sum(x * int(t) for x, t in zip([3600, 60, 1], times.split(":"))) for times in train['time_exit']]
    #1538352000

train['timestamp_entry'] = 1538352000 + train['time_entry_seconds']
train['timestamp_exit'] = 1538352000 + train['time_exit_seconds']

train['timestamp_entry'] = [a + 86400*(int(day.split('_')[1])-1) for a, day in zip(train['timestamp_entry'], train['hash'])]
train['timestamp_exit'] = [a + 86400*(int(day.split('_')[1])-1) for a, day in zip(train['timestamp_exit'], train['hash'])]

train['day_of_week'] = pd.DatetimeIndex(train['timestamp_entry']*10**9).weekday

test['time_entry_seconds'] = [sum(x * int(t) for x, t in zip([3600, 60, 1], times.split(":"))) for times in test['time_entry']] 
test['time_exit_seconds'] = [sum(x * int(t) for x, t in zip([3600, 60, 1], times.split(":"))) for times in test['time_exit']]
    #1538352000

test['timestamp_entry'] = 1538352000 + test['time_entry_seconds']
test['timestamp_exit'] = 1538352000 + test['time_exit_seconds']

test['timestamp_entry'] = [a + 86400*(int(day.split('_')[1])-1) for a, day in zip(test['timestamp_entry'], test['hash'])]
test['timestamp_exit'] = [a + 86400*(int(day.split('_')[1])-1) for a, day in zip(test['timestamp_exit'], test['hash'])]

test['day_of_week'] = pd.DatetimeIndex(test['timestamp_entry']*10**9).weekday

#del train['timestamp_entry'],train['timestamp_exit'],test['timestamp_entry'],test['timestamp_exit']
del train['time_entry_seconds'],train['time_exit_seconds'],test['time_entry_seconds'],test['time_exit_seconds']


#%% using the function to transform the time feature in seconds  
train = split_datetime(train, value = 0)
test = split_datetime(test, value = 0)

train = split_datetime(train, value = 1)
test = split_datetime(test, value = 1)

#%% duration of every tragetory in the base 
train['last_passo_duration'] = train['time_exit']-train['time_entry']
test['last_passo_duration'] = test['time_exit']-test['time_entry']

#%% flag to tragectories that begin in the center 
train['in_center'] = np.where((train['x_entry'] >= lim_esq)&(train['x_entry'] <= lim_dir)&(train['y_entry'] >= lim_inf)&(train['y_entry'] <= lim_sup),1,0 )
test['in_center'] = np.where((test['x_entry'] >= lim_esq)&(test['x_entry'] <= lim_dir)&(test['y_entry'] >= lim_inf)&(test['y_entry'] <= lim_sup),1,0)

#%% percentage of tragetories in center  
aux = pd.DataFrame(train.groupby('hash')['in_center'].mean())
aux.rename(columns={'in_center':'in_center_percentage'}, inplace=True)
train = pd.DataFrame.merge(train, aux, how='left', left_on='hash', right_index=True)

aux = pd.DataFrame(test.groupby('hash')['in_center'].mean())
aux.rename(columns={'in_center':'in_center_percentage'}, inplace=True)
test = pd.DataFrame.merge(test, aux, how='left', left_on='hash', right_index=True)

aux = pd.DataFrame(train.groupby('hash')['in_center'].std())
aux.rename(columns={'in_center':'in_center_std'}, inplace=True)
train = pd.DataFrame.merge(train, aux, how='left', left_on='hash', right_index=True)

aux = pd.DataFrame(test.groupby('hash')['in_center'].std())
aux.rename(columns={'in_center':'in_center_std'}, inplace=True)
test = pd.DataFrame.merge(test, aux, how='left', left_on='hash', right_index=True)


#%% Building valid train and test dataframes

train = train.reset_index()
df_train = train.groupby('hash')['index'].max().reset_index()
train_idx = df_train['index']
df_train = pd.merge(df_train, train, on=['hash','index'], how='left')
train = train[~train['index'].isin(train_idx)]

test = test.reset_index()
df_test = test.groupby('hash')['index'].max().reset_index()
test_idx = df_test['index']
df_test = pd.merge(df_test, test, on=['hash','index'], how='left')
test = test[~test['index'].isin(test_idx)]


#%% catching the result vector of the path  
vetorA = train.groupby('hash')['index'].min().reset_index()
vetorA = pd.merge(vetorA, train, on=['hash','index'], how='left')
vetorA = vetorA[['hash','x_entry','y_entry']]
vetorA.columns = ['hash','xA','yA']
df_train = pd.merge(df_train, vetorA, on='hash', how='left')

vetorB = train.groupby('hash')['index'].max().reset_index()
vetorB = pd.merge(vetorB, train, on=['hash','index'], how='left')
vetorB = vetorB[['hash','x_entry','y_entry']]
vetorB.columns = ['hash','xB','yB']
df_train = pd.merge(df_train, vetorB, on='hash', how='left')

df_train['x_vetor'] = df_train['xB'] - df_train['xA']
df_train['y_vetor'] = df_train['yB'] - df_train['yA']

df_train = df_train.drop(['xA','yA'], axis=1)
df_train = df_train.drop(['xB','yB'], axis=1)

vetorA = test.groupby('hash')['index'].min().reset_index()
vetorA = pd.merge(vetorA, test, on=['hash','index'], how='left')
vetorA = vetorA[['hash','x_entry','y_entry']]
vetorA.columns = ['hash','xA','yA']
df_test = pd.merge(df_test, vetorA, on='hash', how='left')

vetorB = test.groupby('hash')['index'].max().reset_index()
vetorB = pd.merge(vetorB, test, on=['hash','index'], how='left')
vetorB = vetorB[['hash','x_entry','y_entry']]
vetorB.columns = ['hash','xB','yB']
df_test = pd.merge(df_test, vetorB, on='hash', how='left')

df_test['x_vetor'] = df_test['xB'] - df_test['xA']
df_test['y_vetor'] = df_test['yB'] - df_test['yA']

df_test = df_test.drop(['xA','yA'], axis=1)
df_test = df_test.drop(['xB','yB'], axis=1)

#%% catch the result vector between the first point and the center point 
vetorA = train.groupby('hash')['index'].min().reset_index()
vetorA = pd.merge(vetorA, train, on=['hash','index'], how='left')
vetorA = vetorA[['hash','x_entry','y_entry']]
vetorA.columns = ['hash','xA','yA']
df_train = pd.merge(df_train, vetorA, on='hash', how='left')

df_train['x_centro'] = x_centro
df_train['y_centro'] = y_centro

df_train['x_center_vetor'] = df_train['x_centro'] - df_train['xA']
df_train['y_center_vetor'] = df_train['y_centro'] - df_train['yA']

df_train = df_train.drop(['xA','yA'], axis=1)
df_train = df_train.drop(['x_centro','y_centro'], axis=1)

vetorA = test.groupby('hash')['index'].min().reset_index()
vetorA = pd.merge(vetorA, test, on=['hash','index'], how='left')
vetorA = vetorA[['hash','x_entry','y_entry']]
vetorA.columns = ['hash','xA','yA']
df_test = pd.merge(df_test, vetorA, on='hash', how='left')

df_test['x_centro'] = x_centro
df_test['y_centro'] = y_centro

df_test['x_center_vetor'] = df_test['x_centro'] - df_test['xA']
df_test['y_center_vetor'] = df_test['y_centro'] - df_test['yA']

df_test = df_test.drop(['xA','yA'], axis=1)
df_test = df_test.drop(['x_centro','y_centro'], axis=1)

#%% angle between the result vector of the path and the result center vector, to stimate if the tragectory is in the center direction 

top = (df_train['x_vetor']*df_train['x_center_vetor'])+(df_train['y_vetor']*df_train['y_center_vetor'])
down = np.sqrt((df_train['x_vetor']**2) + (df_train['y_vetor']**2))*np.sqrt((df_train['x_center_vetor']**2) + (df_train['y_center_vetor']**2))
df_train['ang_result_center'] = np.cos(pd.DataFrame(top)/pd.DataFrame(down))

top = (df_test['x_vetor']*df_test['x_center_vetor'])+(df_test['y_vetor']*df_test['y_center_vetor'])
down = np.sqrt((df_test['x_vetor']**2) + (df_test['y_vetor']**2))*np.sqrt((df_test['x_center_vetor']**2) + (df_test['y_center_vetor']**2))
df_test['ang_result_center'] = np.cos(pd.DataFrame(top)/pd.DataFrame(down))

#0.6428 = cos of 50°

df_train['going_center'] = np.where(df_train['ang_result_center'] >= 0.7071,1,0)
df_test['going_center'] = np.where(df_test['ang_result_center'] >= 0.7071,1,0)

#%% path begin time
time1 = train.groupby('hash')['index'].min().reset_index()
time1 = pd.merge(time1, train, on=['hash','index'], how='left')
time1 = time1[['hash','time_entry']]
time1.columns = ['hash','trajectory_init_time']
df_train = pd.merge(df_train, time1, on='hash', how='left')

time2 = test.groupby('hash')['index'].min().reset_index()
time2 = pd.merge(time2, test, on=['hash','index'], how='left')
time2 = time2[['hash','time_entry']]
time2.columns = ['hash','trajectory_init_time']
df_test = pd.merge(df_test, time2, on='hash', how='left')

#%% first point of the path
time1 = train.groupby('hash')['index'].min().reset_index()
time1 = pd.merge(time1, train, on=['hash','index'], how='left')
time1 = time1[['hash','x_entry','y_entry']]
time1.columns = ['hash','x_entry_init','y_entry_init']
df_train = pd.merge(df_train, time1, on='hash', how='left')

time2 = test.groupby('hash')['index'].min().reset_index()
time2 = pd.merge(time2, test, on=['hash','index'], how='left')
time2 = time2[['hash','x_entry','y_entry']]
time2.columns = ['hash','x_entry_init','y_entry_init']
df_test = pd.merge(df_test, time2, on='hash', how='left')

#%% travel total duration 
df_train['travel_duration_total'] = df_train['time_exit']-df_train['trajectory_init_time']
df_test['travel_duration_total'] = df_test['time_exit']-df_test['trajectory_init_time']

#%% norm of result vector 
df_train['norma_resultante'] = np.sqrt(df_train['x_vetor']**2 + df_train['y_vetor']**2)
df_test['norma_resultante'] = np.sqrt(df_test['x_vetor']**2 + df_test['y_vetor']**2)

#%% distance from center to last point of path
df_train['dist_ate_centro_ult'] = np.sqrt((x_centro - df_train['x_entry'])**2 + (y_centro - df_train['y_entry'])**2)
df_test['dist_ate_centro_ult'] = np.sqrt((x_centro - df_test['x_entry'])**2 + (y_centro - df_test['y_entry'])**2)

#%% distance from center to first point of path
df_train['dist_ate_centro_prim'] = np.sqrt((x_centro - df_train['x_entry_init'])**2 + (y_centro - df_train['y_entry_init'])**2)
df_test['dist_ate_centro_prim'] = np.sqrt((x_centro - df_test['x_entry_init'])**2 + (y_centro - df_test['y_entry_init'])**2)

#%% aproximation feature, calculated by the diference of the firsth and the last point distance to center
# 1 if aproximated 0 if not 
df_train['diferenca_dist'] = df_train['dist_ate_centro_ult'] - df_train['dist_ate_centro_prim']
df_test['diferenca_dist'] = df_test['dist_ate_centro_ult'] - df_test['dist_ate_centro_prim']

# value of the diference 
df_train['aprox'] = np.where(df_train['diferenca_dist'] >= 0,1,0)
df_test['aprox'] = np.where(df_test['diferenca_dist'] >= 0,1,0)

#%% mean speed of last tragetory
df_train['velocidade_mdia_last_pass'] =df_train['norma_resultante']/df_train['last_passo_duration']
df_test['velocidade_mdia_last_pass'] =df_test['norma_resultante']/df_test['last_passo_duration']

#%% mean speed per tragectory
df_train['velocidade_mdia_trajeto'] =df_train['norma_resultante']/df_train['travel_duration_total']
df_test['velocidade_mdia_trajeto'] =df_test['norma_resultante']/df_test['travel_duration_total']


#%% dropping some columns that we dont use
df_train = df_train.drop('hash', axis = 1)
df_train = df_train.drop('trajectory_id', axis = 1)

test = test.drop('hash', axis = 1)
test = test.drop('trajectory_id', axis = 1)

train = train.drop('hash', axis = 1)
train = train.drop('trajectory_id', axis = 1)

sub = pd.DataFrame()
sub['id'] = df_test['trajectory_id']
df_test = df_test.drop('hash', axis = 1)
df_test = df_test.drop('trajectory_id', axis = 1)


#%% target calculation and input
df_train['target'] = 0
df_train['target'][(df_train['x_exit'] >= lim_esq)&(df_train['x_exit'] <= lim_dir)&(df_train['y_exit'] >= lim_inf)&(df_train['y_exit'] <= lim_sup)] = 1

#%%
del df_train['index'], df_test['index']
target = df_train['target']
del df_train['target'], df_train['x_exit'], df_train['y_exit'], df_test['x_exit'], df_test['y_exit']
features = [c for c in df_train.columns if c not in ['target']]

#%% Model training and prediction
folds = KFold(n_splits=5, shuffle=False, random_state=seed)
oof_preds = np.zeros(df_train.shape[0])
sub_preds = np.zeros(df_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(target, target)):
    trn_x, trn_y = df_train[features].loc[trn_], target.loc[trn_]
    val_x, val_y = df_train[features].loc[val_], target.loc[val_]
    
    clf = train_lgb(seed, lgb.Dataset(trn_x,label = trn_y), lgb.Dataset(val_x,label = val_y))
#    lgb.plot_importance(clf, importance_type='split', max_num_features=20)
#    lgb.plot_importance(clf, importance_type='gain', max_num_features=20)
    oof_preds[val_] = clf.predict(val_x)
    sub_preds += clf.predict(df_test[features])/ 5
    
    print('Fold  f1 : %.6f' % f1_score(val_y, np.round(oof_preds[val_])))
    
print('Full f1 score %.6f' % f1_score(target, np.round(oof_preds)))  

    
 
#%% submission CSV
print("Creating submission")
sub_predsfinal = np.round(sub_preds)
sub['target'] = sub_predsfinal
sub.to_csv("G:\\Meu Drive\\LP&D\\NextWave\\submission.csv", index=False)








