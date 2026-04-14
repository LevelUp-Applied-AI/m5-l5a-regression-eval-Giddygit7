import importlib.util

import os

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lab_regression.py'))

spec = importlib.util.spec_from_file_location("lab_regression_real", file_path)

module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(module)

load_data = module.load_data

split_data = module.split_data

build_logistic_pipeline = module.build_logistic_pipeline

build_ridge_pipeline = module.build_ridge_pipeline

evaluate_classifier = module.evaluate_classifier

evaluate_regressor = module.evaluate_regressor

run_cross_validation = module.run_cross_validation