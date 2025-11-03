#!/usr/bin/env python
# coding: utf-8

# # 6. Decision Trees and Ensemble Learning
# This week, we'll talk about decision trees and tree-based ensemble algorithms

# In[ ]:





# # 6.1 Credit risk scoring project

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# # 6.2 Data cleaning and preparation
# - Downloading the dataset
# - Re-encoding the categorical variables
# - Doing the train/validation/test split

# In[2]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv'


# In[3]:


get_ipython().system('wget $data')


# In[4]:


get_ipython().system('head CreditScoring.csv')


# In[5]:


df = pd.read_csv('CreditScoring.csv')


# In[6]:


df.head()


# In[7]:


df.columns = df.columns.str.lower()


# In[8]:


status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)


# In[9]:


df.head()


# In[10]:


home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)


# In[11]:


df.head()


# In[12]:


# 99999999.0 is a missing value in this data set, so we identify where we might have these.
df.describe().round()


# In[13]:


columns = ['income', 'assets', 'debt']
for col in columns:
    df[col] = df[col].replace(to_replace=99999999.0, value=np.nan)


# In[14]:


# Let's check that we no longer have 99999999.0 as missing value
df.describe().round()


# In[15]:


# drop the 1 'unk' record but also reset the index so we don't have a missing row number.
df = df[df.status != 'unk'].reset_index(drop=True)


# In[16]:


from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)


# In[17]:


len(df_test), len(df_train), len(df_val)


# In[18]:


df_train


# In[19]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[20]:


y_train = (df_train.status == 'default').astype('int').values
y_val = (df_val.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values


# In[21]:


del df_train['status']
del df_val['status']
del df_test['status']


# In[22]:


df_train


# # 6.3 Decision trees
# - How a decision tree looks like
# - Training a decision tree
# - Overfitting
# - Controlling the size of a tree

# In[23]:


def asses_risk(client):
    if client['records'] == 'yes':
        if client['job'] == 'parttime':
            return 'default'
        else:
            return 'ok'
    else:
        if client['assets'] > 6000:
            return 'ok'
        else:
            return 'default'


# In[24]:


df_train.iloc[0]


# In[25]:


xi = df_train.iloc[0].to_dict()


# In[26]:


asses_risk(xi)


# In[71]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# In[28]:


train_dicts = df_train.fillna(0).to_dict(orient='records')


# In[29]:


dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)


# In[30]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train) 


# In[31]:


val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)


# In[32]:


y_pred = dt.predict_proba(X_val)[:,1]


# In[33]:


roc_auc_score(y_val, y_pred)


# In[34]:


y_pred = dt.predict_proba(X_train)[:,1]
roc_auc_score(y_train, y_pred)


# In[35]:


dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)


# In[36]:


# a decision tree with max depth = 2 is better than the decision tree with no depth limit.

y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train:', auc)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('val:', auc) 


# In[37]:


from sklearn.tree import export_text

print(export_text(dt, feature_names=list(dv.get_feature_names_out())))


# # 6.4 Decision tree learning algorithm
# - Finding the best split for one column
# - Finding the best split for the entire dataset
# - Stopping criteria
# - Decision tree learning algorithm

# In[38]:


data = [
    [8000, 'default'],
    [2000, 'default'],
    [   0, 'default'],
    [5000, 'ok'],
    [5000, 'ok'],
    [4000, 'ok'],
    [9000, 'ok'],
    [3000, 'default'],
]

df_example = pd.DataFrame(data, columns=['assets', 'status'])
df_example


# In[39]:


df_example.sort_values('assets')


# In[40]:


Ts = [0, 2000, 3000, 4000, 5000, 8000]


# In[41]:


T = 4000
df_left = df_example[df_example.assets <= T]
df_right = df_example[df_example.assets > T]

display(df_left)
print(df_left.status.value_counts(normalize=True))
display(df_right)
print(df_left.status.value_counts(normalize=True))


# In[42]:


from IPython.display import display


# In[43]:


for T in Ts:
    print(T)
    df_left = df_example[df_example.assets <= T]
    df_right = df_example[df_example.assets > T]

    display(df_left)
    print(df_left.status.value_counts(normalize=True))
    display(df_right)
    print(df_right.status.value_counts(normalize=True))

    print()


# In[44]:


# adding a simulated debit column
data = [
    [8000, 3000, 'default'],
    [2000, 1000, 'default'],
    [   0, 1000, 'default'],
    [5000, 1000, 'ok'],
    [5000, 1000, 'ok'],
    [4000, 1000, 'ok'],
    [9000,  500, 'ok'],
    [3000, 2000, 'default'],
]

df_example = pd.DataFrame(data, columns=['assets', 'debt', 'status'])
df_example


# In[45]:


# split: 500, 1000, 2000
df_example.sort_values('debt')


# In[46]:


thresholds = {
    'assets': [0, 2000, 3000, 4000, 5000, 8000], 
    'debt': [500, 1000, 2000]
}


# In[47]:


for feature, Ts in thresholds.items():
    print('######################')
    print(feature)
    for T in Ts:
        print(T)
        df_left = df_example[df_example[feature] <= T]
        df_right = df_example[df_example[feature] > T]

        display(df_left)
        print(df_left.status.value_counts(normalize=True))
        display(df_right)
        print(df_right.status.value_counts(normalize=True))

        print('######################')


# # 6.5 Decision trees parameter tuning
# - selecting max_depth
# - selecting min_samples_leaf

# In[48]:


for d in [1,2,3,4,5,6,10,15,20,None]:
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    print('%4s => %.3f' % (d, auc))


# In[49]:


scores = []
for d in [4, 5, 6, 7, 10, 15, 20, None]:
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, s, auc))        


# In[50]:


columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.head()


# In[51]:


df_scores.sort_values(by='auc', ascending=False).head()


# In[52]:


df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'],
                values=['auc'])

df_scores_pivot.round(3)


# In[53]:


sns.heatmap(df_scores_pivot, annot=True, fmt='.3f')


# In[54]:


df = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
df.fit(X_train, y_train)


# # 6.6 Ensembles and random forest
# - Board of experts
# - Ensembling models
# - Random forest - ensembling decision trees
# - Tuning random forest

# In[64]:


from sklearn.ensemble import RandomForestClassifier


# In[83]:


rf = RandomForestClassifier(n_estimators=10, random_state=1)
rf.fit(X_train, y_train)


# In[77]:


y_pred = rf.predict_proba(X_val)[:, 1]
y_pred[:10]


# In[75]:


roc_auc_score(y_val, y_pred)


# In[85]:


# by specifying a random_state we can ensure we get the same result everytime we run the model
rf.predict_proba(X_val[[0]])


# In[87]:


scores = []

for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    scores.append((n, auc))


# In[89]:


df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
df_scores


# In[90]:


# after about 50 estimators (trees) the model doesn't get much more accurate. 
plt.plot(df_scores.n_estimators, df_scores.auc)


# In[92]:


# lets try training a RF model with different max depth and min-leaf parameters

scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)

        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))


# In[93]:


columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.head()


# In[94]:


for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.auc, 
             label='max_depth=%d' % d)
plt.legend()


# In[95]:


# 10 looks like the best max_depth. Now let's move on to min-leaf
max_depth = 10


# In[97]:


# lets try training a RF model with different min-leaf parameters

scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=10,
                                    min_samples_leaf=s,
                                    random_state=1)

        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))


# In[98]:


columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.head()


# In[106]:


# 3/blue appears to be the best earliest and longest.

colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)
plt.legend()


# In[107]:


min_sample_leaf = 3


# In[108]:


rf = RandomForestClassifier(n_estimators=n,
                            max_depth=max_depth,
                            min_samples_leaf=min_sample_leaf,
                            random_state=1, 
                            n_jobs=-1)

rf.fit(X_train, y_train)


# #### Other useful parametes:
# - max_features
# - bootstrap
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# # 6.7 Gradient boosting and XGBoost
# - Gradient boosting vs random forest
# - Installing XGBoost
# - Training the first model
# - Performance monitoring
# - Parsing xgboost's monitoring output

# In[109]:


get_ipython().system('pip install xgboost')


# In[111]:


import xgboost as xgb


# In[115]:


features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


# In[120]:


xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 0, 
}
model = xgb.train(xgb_params, dtrain, num_boost_round=10)


# In[121]:


y_pred = model.predict(dval)


# In[122]:


roc_auc_score(y_val, y_pred)


# In[124]:


watchlist = [(dtrain, 'train'), (dval, 'val')]


# In[128]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3,\n    'max_depth': 6,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n\n    'seed': 1,\n    'verbosity': 1, \n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                   evals=watchlist)\n")


# In[129]:


print(output.stdout)


# In[130]:


print(output.stdout)


# In[134]:


s = output.stdout


# In[135]:


line = s.split('\n')[0]


# In[136]:


num_iter, train_auc, val_auc = line.split('\t')


# In[137]:


num_iter, train_auc, val_auc


# In[138]:


int(num_iter.strip('[]'))


# In[141]:


float(train_auc.split(':')[1])


# In[145]:


float(val_auc.split(':')[1])


# In[148]:


def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))

    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results


# In[149]:


df_score = parse_xgb_output(output)


# In[152]:


# the model is over fitting

plt.plot(df_score.num_iter, df_score.train_auc, label='train')
plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend()


# In[154]:


# lets look at just val 
plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend()


# # 6.8 XGBoost parameter tuning
# Tuning the following parameters:
# 
# - eta
# - max_depth
# - min_child_weight

# In[ ]:


scores = {}


# In[189]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': .1, \n    'max_depth': 6,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[190]:


key = 'eta=%s' % (xgb_params['eta'])


# In[214]:


scores[key] = parse_xgb_output(output)
key


# In[211]:


scores.keys()


# In[212]:


scores['eta=1'].head(3)


# In[227]:


# 0.1 is the best.
etas = ['eta=0.1', 'eta=0.3', 'eta=0.01']

for eta in etas:
    df_score = scores[eta]
    plt.plot(df_score.num_iter, df_score.val_auc, label=eta)

plt.legend()


# In[228]:


eta = 0.1


# In[229]:


scores = {}


# In[236]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1, \n    'max_depth': 10,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[237]:


key = 'max_depth=%s' % (xgb_params['max_depth'])
scores[key] = parse_xgb_output(output)
key


# In[240]:


scores.keys()


# In[242]:


# max_depth 3 is best
for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)

plt.legend()


# In[243]:


del scores['max_depth=10']


# In[244]:


# max_depth 3 is best
for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)

plt.ylim(0.8, 0.84)
plt.legend()


# In[ ]:


# min child weight tuning


# In[245]:


scores = {}


# In[250]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1, \n    'max_depth': 3,\n    'min_child_weight': 13,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[251]:


key = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
scores[key] = parse_xgb_output(output)
key


# In[252]:


scores.keys()


# In[254]:


for min_child_weight, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=min_child_weight)

plt.ylim(0.82, 0.84)
plt.legend()


# In[255]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1, \n    'max_depth': 3,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=175,\n                  evals=watchlist)\n")


# Other parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
# 
# Useful ones:
# - `subsample` and `colsample_bytree`
# - `lambda` and `alpha`

# # 6.9 Selecting the final model
# - Choosing between xgboost, random forest and decision tree
# - Training the final model
# - Saving the model

# In[256]:


# our best decision tree model
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)


# In[257]:


y_pred = dt.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)


# In[258]:


# our best RF model 
rf = RandomForestClassifier(n_estimators=200,
                            max_depth=10,
                            min_samples_leaf=3,
                            random_state=1)
rf.fit(X_train, y_train)


# In[259]:


y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)


# In[260]:


xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=175)


# In[262]:


y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)


# In[263]:


df_full_train = df_full_train.reset_index(drop=True)


# In[264]:


y_full_train = (df_full_train.status == 'default').astype(int).values


# In[265]:


y_full_train


# In[266]:


del df_full_train['status']


# In[269]:


dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)


# In[284]:


feat_names = dv.get_feature_names_out().astype(str).tolist()

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=feat_names)

dtest = xgb.DMatrix(X_test, feature_names=feat_names)


# In[285]:


xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=175)


# In[286]:


y_pred = model.predict(dtest)


# In[287]:


# model generalized quite well to unseen data. We didn't actually overfit!
roc_auc_score(y_test, y_pred)


# # 6.10 Summary
# - Decision trees learn if-then-else rules from data.
# - Finding the best split: select the least impure split. This algorithm can overfit, that's why we control it by limiting the max depth and the size of the group.
# - Random forest is a way of combininig multiple decision trees. It should have a diverse set of models to make good predictions.
# - Gradient boosting trains model sequentially: each model tries to fix errors of the previous model. XGBoost is an implementation of gradient boosting.
# 

# # 6.11 Explore more
# - For this dataset we didn't do EDA or feature engineering. You can do it to get more insights into the problem.
# - For random forest, there are more parameters that we can tune. Check max_features and bootstrap.
# - There's a variation of random forest caled "extremely randomized trees", or "extra trees". Instead of selecting the best split among all possible thresholds, it selects a few thresholds randomly and picks the best one among them. Because of that extra trees never overfit. In Scikit-Learn, they are implemented in ExtraTreesClassifier. Try it for this project.
# - XGBoost can deal with NAs - we don't have to do fillna for it. Check if not filling NA's help improve performance.
# - Experiment with other XGBoost parameters: subsample and colsample_bytree.
# - When selecting the best split, decision trees find the most useful features. This information can be used for understanding which features are more important than otheres. See example here for random forest (it's the same for plain decision trees) and for xgboost
# - Trees can also be used for solving the regression problems: check DecisionTreeRegressor, RandomForestRegressor and the objective=reg:squarederror parameter for XGBoost.

# In[ ]:




