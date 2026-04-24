# Tracking Config
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.sklearn.autolog()

# wrapper for MLflow logging
def run_mlflow_experiment(model_trainer, X_train, y_train, X_test, y_test, run_name="Model Experiment"):
    """
    Trains, evaluates, and logs a model to MLflow.
    
    Args:
        model_trainer: A function that trains and returns a model.
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
        run_name: String name for the MLflow run.
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Train model
        model = model_trainer(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        # evaluate performance
        eval_metrics, _ = evaluate_model_numerics(model, X_test, y_test)
        mlflow.log_metrics(eval_metrics)
        # Analyze residuals
        predictions = model.predict(X_test)
        res_metrics = residuals_analysis_numerics(y_test, predictions)
        mlflow.log_metrics(res_metrics)
        # closing
        run_id = run.info.run_id
        print(f"Successfully logged '{run_name}' under Run ID: {run_id}")
        return model, run_id, eval_metrics, res_metrics

# Usage:
# model, rid = run_mlflow_experiment(train_decision_tree, X_train, y_train, X_test, y_test, "Decision Tree Regressor")


import os

# Hyper parameter tuning
# Experiment Setup
experiment_name = "Decision_Tree_Hyperparameter_Tuning"
artifact_uri = f"file://{os.path.join(os.getcwd(), 'mlruns_artifacts')}"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name, artifact_location=artifact_uri)

mlflow.set_experiment(experiment_name)

    # Depth:
results = []
# Example: Testing different depths for your Decision Tree
for i in [1, 2, 3, 4, 5]:
    # Define a custom trainer target hyperparameter
    def train_dt_with_depth(X, y):
        model = DecisionTreeRegressor(max_depth=i, random_state=42)
        model.fit(X, y)
        return model

    # Run the experiment
    model, rid, eval_met, res_met = run_mlflow_experiment(
        train_dt_with_depth, 
        X_train, y_train, 
        X_val, y_val, 
        run_name=f"DT_Depth_{i}"
    )
    
    # Track results (pseudo-code)
    results.append({
        "max_depth": i,
        "run_id": rid,
        "rmse": eval_met['rmse'],
        "skew": res_met['res_skew']
    })
