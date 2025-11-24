import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_X_y

if not hasattr(BaseEstimator, '_validate_data'):
    def _validate_data(self, X, y=None, reset=True, **check_params):
        if y is None:
            X_checked = check_array(X, **check_params)
            if hasattr(X_checked, 'shape'):
                self.n_features_in_ = X_checked.shape[1]
            return X_checked
        X_checked, y_checked = check_X_y(X, y, **check_params)
        if hasattr(X_checked, 'shape'):
            self.n_features_in_ = X_checked.shape[1]
        return X_checked, y_checked
    BaseEstimator._validate_data = _validate_data

from gplearn.genetic import SymbolicRegressor

DATA_PATH = 'auto-lpkm.tsv'
data = pd.read_csv(DATA_PATH, sep='\t', header=0)
X = data.to_numpy()[:, :-1]
y = data.to_numpy()[:, -1]
feature_names = data.drop('l100km', axis=1).columns.values
X = StandardScaler().fit_transform(X)

BASE_FUNCTIONS = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'inv')
search_space = ParameterGrid({
    'init_depth': [(2, 6), (2, 8)],
    'const_range': [(-10, 10), (-5, 5)],
    'function_set': [
        BASE_FUNCTIONS,
        BASE_FUNCTIONS + ('sin', 'cos'),
    ],
    'stopping_criteria': [0.0, 0.01],
})
mutation_schedules = [
    {'p_crossover': 0.75, 'p_subtree_mutation': 0.10, 'p_point_mutation': 0.10},
    {'p_crossover': 0.70, 'p_subtree_mutation': 0.20, 'p_point_mutation': 0.05},
    {'p_crossover': 0.65, 'p_subtree_mutation': 0.15, 'p_point_mutation': 0.15},
]

common_params = dict(
    population_size=200,
    generations=20,
    metric='mse',
    max_samples=1,
    feature_names=feature_names,
    random_state=42,
    verbose=0,
)

results = []
for grid_params in search_space:
    for mutation in mutation_schedules:
        params = common_params.copy()
        params.update(grid_params)
        params.update(mutation)
        model = SymbolicRegressor(**params)
        y_pred = cross_val_predict(model, X, y, cv=5)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        results.append({
            'params': params,
            'MSE': mse,
            'R2': r2,
        })

best = sorted(results, key=lambda d: (d['MSE'], -d['R2']))[0]
print('Best params:')
for k, v in best['params'].items():
    print(f"{k}: {v}")
print(f"Best MSE: {best['MSE']:.6f}")
print(f"Best R2: {best['R2']:.6f}")
