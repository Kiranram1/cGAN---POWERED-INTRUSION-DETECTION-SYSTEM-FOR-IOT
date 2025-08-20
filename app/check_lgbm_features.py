import joblib

# Load the trained LightGBM model
model = joblib.load('lightGBM_multiclass_classifier.joblib')

# Print feature names and number of features expected
if hasattr(model, 'feature_name_'):
    print('Model expects these features:')
    for i, feat in enumerate(model.feature_name_):
        print(f'{i+1}: {feat}')
    print(f'Number of features expected by model: {len(model.feature_name_)}')
else:
    print('Model feature names not found. Model type:', type(model))

# Optionally, print n_features_in_ if available
if hasattr(model, 'n_features_in_'):
    print(f'n_features_in_: {model.n_features_in_}') 