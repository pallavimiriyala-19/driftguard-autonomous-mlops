import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from driftguard.orchestrator import DriftGuardOrchestrator, ModelConfig, DataConfig, RemediationPlan

# --- 1. Define Mock Model Prediction Function ---
def mock_model_predict(df: pd.DataFrame) -> np.ndarray:
    """ 
    A simple mock prediction function. 
    Assumes features 'feature_a', 'feature_b', 'feature_c' exist.
    """
    if not all(f in df.columns for f in ['feature_a', 'feature_b', 'feature_c']):
        raise ValueError("Missing required features for mock_model_predict")
    
    # Simulate a linear model with some noise
    predictions = (
        0.5 * df['feature_a'] + 
        1.2 * df['feature_b'] - 
        0.3 * df['feature_c'] + 
        np.random.normal(0, 0.1, len(df))
    )
    return predictions.values

# --- 2. Define Mock Retraining Pipeline Callback ---
def mock_retrain_pipeline(plan: RemediationPlan):
    """
    A mock function to simulate triggering an MLOps retraining pipeline.
    In a real scenario, this would call an API, trigger a CI/CD pipeline, etc.
    """
    print(f"\n[MOCK MLOPS] Initiating retraining for model '{plan.model_id}'...")
    print(f"[MOCK MLOPS] Diagnosis: {plan.diagnosis}")
    print(f"[MOCK MLOPS] Recommended action: {plan.action}")
    # Simulate pipeline execution time
    time.sleep(3)
    print(f"[MOCK MLOPS] Retraining pipeline completed for model '{plan.model_id}'. New model version deployed!\n")

# --- 3. Generate Baseline Data ---
def generate_baseline_data(num_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(num_samples)],
        'feature_a': np.random.rand(num_samples) * 10,
        'feature_b': np.random.rand(num_samples) * 5 + 2,
        'feature_c': np.random.normal(0, 1, num_samples),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], num_samples),
        'target': np.random.rand(num_samples) * 100 # Example target, not used by KSDrift
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- 4. Generate Simulated Production Data with Optional Drift ---
def generate_simulated_data_batch(batch_size: int = 50, introduce_drift: bool = False) -> pd.DataFrame:
    current_time = datetime.now()
    data = {
        'timestamp': [current_time - timedelta(seconds=i) for i in range(batch_size)],
        'feature_a': np.random.rand(batch_size) * 10,
        'feature_b': np.random.rand(batch_size) * 5 + 2,
        'feature_c': np.random.normal(0, 1, batch_size),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], batch_size),
        'target': np.random.rand(batch_size) * 100
    }

    df = pd.DataFrame(data)

    if introduce_drift:
        print("\n--- INTRODUCING ARTIFICIAL DRIFT! ---")
        # Introduce a shift in feature_a distribution
        df['feature_a'] = df['feature_a'] * 1.5 + 5 # Mean shift, increased variance
        # Introduce a shift in feature_c distribution
        df['feature_c'] = df['feature_c'] + 3 # Mean shift
        # Introduce a new category in categorical_feature
        df['categorical_feature'] = np.random.choice(['A', 'B', 'C', 'D'], batch_size, p=[0.2, 0.2, 0.2, 0.4])
        print("-------------------------------------\n")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- 5. Main Execution Script ---
if __name__ == "__main__":
    print("Starting DriftGuard Autonomous MLOps Agent Demo...")

    # Configuration
    model_config = ModelConfig(
        model_id="customer_churn_predictor_v1",
        features=['feature_a', 'feature_b', 'feature_c'], # 'categorical_feature' needs special handling for KSDrift
        target_column="target"
    )

    data_config = DataConfig(
        timestamp_column="timestamp",
        batch_size=20, # How many samples to accumulate before a monitoring check
        history_size=200, # How many recent samples to keep in history for drift detection
        monitoring_interval_seconds=10 # Check every 10 seconds
    )

    # Generate baseline data
    df_baseline = generate_baseline_data(num_samples=500)
    print(f"Generated baseline data with {len(df_baseline)} samples.")

    # Initialize the Orchestrator
    orchestrator = DriftGuardOrchestrator(
        model_config=model_config,
        data_config=data_config,
        baseline_data=df_baseline, 
        prediction_function=mock_model_predict, 
        retrain_callback=mock_retrain_pipeline
    )

    # Start the orchestrator (this starts all agents in separate threads)
    orchestrator.start()

    num_batches_to_simulate = 15 # Total batches to run
    batch_counter = 0
    try:
        while batch_counter < num_batches_to_simulate:
            print(f"\n--- Simulating Production Batch {batch_counter + 1}/{num_batches_to_simulate} ---")
            # Introduce drift after a few batches
            introduce_drift_flag = batch_counter >= 5
            
            new_production_data = generate_simulated_data_batch(
                batch_size=data_config.batch_size,
                introduce_drift=introduce_drift_flag
            )
            
            orchestrator.process_data_batch(new_production_data)
            batch_counter += 1
            time.sleep(data_config.monitoring_interval_seconds / 2) # Simulate data arriving asynchronously

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        orchestrator.shutdown()
        print("DriftGuard Demo Finished.")

