import os
import tarfile
import urllib
import matplotlib.pyplot as plt

import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

####데이터 불러오기
DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()  #첫 다섯 행 확인
housing.info()  #전체 행 수, 각 특성의 타입과 널이 아닌 값의 개수 확인

housing['ocean_proximity'].value_counts()   #카테고리 종류, 각 개수

housing.describe()  #숫자형 특성의 요약 정보

housing.hist(bins=50, figsize=(20,15))  #모든 숫자형 특성에 대한 히스토그램 생성
plt.show()


#### 테스트 세트 만들기
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)

from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # index 열이 추가된 데이터프레임이 반환
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']   # 고유 식별자를 만들어줌
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels = [1,2,3,4,5])

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)   # 소득 카테고리를 기반으로 계층 샘플링
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set['income_cat'].value_counts() / len(strat_test_set)   # 소득 카테고리의 비율

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)


#### 데이터 탐색과 시각화 (EDA)
housing = strat_train_set.copy()    # 훈련 세트를 손상시키지 않기 위해 복사본을 만들어 사용
### 지리적 데이터 시각화
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.show()

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100,
             label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'),
             colorbar=True, sharex=False)
plt.legend()
plt.show()
### 상관관계 조사
# corr_matrix = housing.corr()

# from pandas.plotting import scatter_matrix
# attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
# scatter_matrix(housing[attributes], figsize=(12,8))
# plt.show()
#
# housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
# plt.show()
### 특성 조합으로 실험
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']


#### 데이터 정제
housing_labels = strat_train_set['median_house_value'].copy()
housing = strat_train_set.drop('median_house_value', axis=1)

# 값이 없는 total_bedrooms을 정제
housing.dropna(subset=['total_bedrooms'])   #옵션1 - 해당 구역 제거
housing.drop('total_bedrooms', axis=1)      #옵션2 - 전체 특성 삭제
median = housing['total_bedrooms'].median() #옵션3 - 중간값으로 채움
housing['total_bedrooms'].fillna(median, inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)
imputer.statistics_ # 계산된 중간값이 저장되어 있음
housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


#### 텍스트와 범주형 특성 다루기
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)

# 텍스트에서 숫자로 변환
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

ordinal_encoder.categories_ #카테고리 리스트를 얻을 수 있음

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

housing_cat_1hot.toarray()
##or
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

cat_encoder.categories_

#### 나만의 변환기
# 추가 특성을 위해 사용자 정의 변환기를 만듦
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6    # 열 인덱스
#or
col_names = 'total_rooms', 'total_bedrooms', 'population', 'households'
rooms_ix, bedrooms_ix, population_ix, households_ix = [housing.columns.get_loc(c) for c in col_names]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # *args or **kaege 없음
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # 아무것도 하지 않음
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, bedrooms_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.to_numpy())    # 기존 데이터에 새로운 열 2개 추가됨

# housing_extra_attribs는 넘파이 배열이기 때문에 열 이름이 없음. -> 데이터 프레임으로 복원
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns = list(housing.columns) + ['rooms_per_household', 'population_per_household'],
    index = housing.index)
housing_extra_attribs.head()

#### 변환 파이프라인
# 수치형 특성을 전처리하기 위해 파이프라인 만듦
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler()),
                        ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr

# 범주형과 수치형을 한 번에 처리
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([ ('num', num_pipeline, num_attribs),
                                    ('cat', OneHotEncoder(), cat_attribs),
                                    ])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape

#### 훈련 세트에서 훈련 & 평가
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 훈련 샘플 몇 개를 사용해 전체 파이프라인을 적용
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print('예측:', lin_reg.predict(some_data_prepared))
print('레이블:', list(some_labels))

# rmse 측정
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse    # 과소적합

# decisionTreeRegressor 사용
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse   #과대적합

#### 교차 검증을 사용한 평가
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('score:', scores)
    print('avg:', scores.mean())
    print('std:', scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring='neg_mean_squared_error',cv=10)
lin_rms_scores = np.sqrt(-lin_scores)
display_scores(lin_rms_scores)

# 앙상블
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
pd.Series(np.sqrt(-scores)).describe()

# SVM
from sklearn.svm import SVR
svm_reg = SVR(kernel='linear')
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse

#### 그리드 탐색
from sklearn.model_selection import GridSearchCV

param_grid = [
    # 12(=3*4)개의 하이퍼파라미터 조합을 시도함
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    # bootstrap은 False로 하고 6(=2*3)개의 조합을 시도
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# 다섯 개의 폴드로 훈련하면 총 (12+6)*5=90번의 훈련이 일어남
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# 최상의 파라미터 조합은 다음과 같음
grid_search.best_params_
grid_search.best_estimator_

# 그리드서치에서 테스트한 하이퍼파라미터 조합의 점수를 확인
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

pd.DataFrame(grid_search.cv_results_)

#### 랜덤 탐색
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


#### 최상의 모델과 오차 분석
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

#### 테스트 세트로 시스템 평가하기
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# 신뢰구간 계사
from scipy import stats
confidence = 0.95
squared_error = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_error) - 1, loc = squared_error.mean(), scale=stats.sem(squared_error)))

#### 연습 문제
#1
from sklearn.model_selection import GridSearchCV

param_grid = [{'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(housing_prepared, housing_labels)

negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse

grid_search.best_params_

#2
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

param_distribs = {'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),}

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions = param_distribs, n_iter=50,
                                cv=5, scoring='neg_mean_squared_error', verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

negative_mse = rnd_search.best_score_
rms = np.sqrt(-negative_mse)
rnd_search.best_params_

expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('Exponential distribution (scale=1.0)')
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title('Log of this distribution')   #원하는 스케일이 정확이 무엇인지 모를 때 사용하면 좋음
plt.hist(np.log(samples), bins=50)
plt.show()

reciprocal_distrib = reciprocal(20, 20000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('Reciprocal distribution (scale=1.0)')
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title('Log og this distribution')
plt.hist(np.log(samples), bins=50)
plt.show()

#3
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

k = 5
top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices

np.array(attributes)[top_k_feature_indices]
sorted(zip(feature_importances, attributes), reverse=True)[:k]

preparation_and_feature_selection_pipline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])
housing_prepared_top_k_features = preparation_and_feature_selection_pipline.fit_transform(housing)

housing_prepared_top_k_features[0:3]
housing_prepared[0:3, top_k_feature_indices]

#4
prepare_select_and_predict_pipline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])

prepare_select_and_predict_pipline.fit(housing, housing_labels)

some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print('Predictions:\t', prepare_select_and_predict_pipline.predict(some_data))
print('Labels:\t\t', list(some_labels))


