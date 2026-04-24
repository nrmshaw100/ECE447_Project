mlflow.sklearn.autolog()
# Enable MLflow autologging

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
        # evaluate performance
        evaluate_model_numerics(model, X_test, y_test)
        # Analyze residuals
        predictions = model.predict(X_test)
        residuals_analysis_numerics(y_test, predictions)
        # closing
        run_id = run.info.run_id
        print(f"Successfully logged '{run_name}' under Run ID: {run_id}")
        return model, run_id

# Usage:
# model, rid = run_mlflow_experiment(train_decision_tree, X_train, y_train, X_test, y_test, "Decision Tree Regressor")


# Hyper parameter tuning

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
    model, rid = run_mlflow_experiment(
        train_dt_with_depth, 
        X_train, y_train, 
        X_val, y_val, 
        run_name=f"DT_Depth_{i}"
    )
    
    # Track results (pseudo-code)
    results.append({
        "max_depth": i,
        "run_id": rid,
        "rmse": eval_metrics['rmse'],
        "skew": res_metrics['res_skew']
    })


