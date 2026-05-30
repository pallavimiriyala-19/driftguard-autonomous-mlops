import time
import uuid
import threading
from collections import deque
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from alibi_detect.cd import KSDrift
from alibi_detect.cd.base import BaseDriftDetector

# --- 1. Data Models and Configurations ---

class ModelConfig(BaseModel):
    model_id: str
    features: List[str]
    target_column: Optional[str] = None # For concept drift, if applicable

class DataConfig(BaseModel):
    timestamp_column: str
    batch_size: int = 100
    history_size: int = 1000 # Number of past records to keep for monitoring
    monitoring_interval_seconds: int = 60 # How often the MonitoringAgent runs

class DriftAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    timestamp: float = Field(default_factory=time.time)
    drift_type: str # e.g., 'data_drift', 'concept_drift'
    drifted_features: List[str] = [] # Features identified with drift
    drift_score: float
    severity: str # 'low', 'medium', 'high'
    details: Dict[str, Any] = {}

class RemediationPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str
    model_id: str
    timestamp: float = Field(default_factory=time.time)
    action: str # e.g., 'retrain', 'notify_human', 'adjust_thresholds'
    diagnosis: str
    recommended_data_range: Optional[Dict[str, Any]] = None # e.g., {'start': '2026-05-29', 'end': '2026-05-30'}
    priority: str # 'critical', 'high', 'medium', 'low'


# --- 2. Agent Base Class and Message Bus (Simplified) ---

class Agent:
    def __init__(self, name: str, message_bus: 'MessageBus'):
        self.name = name
        self.message_bus = message_bus
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self.thread.start()
        print(f"Agent {self.name} started.")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        print(f"Agent {self.name} stopped.")

    def _run(self):
        raise NotImplementedError

class MessageBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self._queue: deque[tuple[str, Any]] = deque()
        self._lock = threading.Lock()
        self._event = threading.Event()

    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def publish(self, topic: str, message: Any):
        with self._lock:
            self._queue.append((topic, message))
            self._event.set() # Signal that there's a new message

    def _process_messages(self):
        while True:
            self._event.wait()
            with self._lock:
                messages_to_process = list(self._queue)
                self._queue.clear()
                self._event.clear()

            for topic, message in messages_to_process:
                if topic in self._subscribers:
                    for callback in self._subscribers[topic]:
                        try:
                            callback(message)
                        except Exception as e:
                            print(f"Error processing message for topic {topic} by {callback.__name__}: {e}")
            time.sleep(0.01) # Small delay to prevent busy-waiting if no messages


# --- 3. Core Agents ---

class DataStore:
    """ In-memory data store for recent production data and baseline. """
    def __init__(self, history_size: int):
        self.production_data_history: deque[pd.DataFrame] = deque(maxlen=history_size)
        self.baseline_data: Optional[pd.DataFrame] = None

    def add_production_batch(self, df: pd.DataFrame):
        self.production_data_history.append(df)

    def get_recent_production_data(self, num_batches: int = 1) -> pd.DataFrame:
        if not self.production_data_history:
            return pd.DataFrame()
        # Concatenate up to num_batches, ensuring not to exceed history size
        recent_dfs = list(self.production_data_history)[-num_batches:]
        return pd.concat(recent_dfs, ignore_index=True) if recent_dfs else pd.DataFrame()

    def set_baseline(self, df: pd.DataFrame):
        self.baseline_data = df

class DataIngestionAgent(Agent):
    def __init__(self, name: str, message_bus: MessageBus, data_store: DataStore, model_config: ModelConfig):
        super().__init__(name, message_bus)
        self.data_store = data_store
        self.model_config = model_config

    def process_incoming_data(self, df: pd.DataFrame):
        """ External entry point for new production data. """
        features_df = df[self.model_config.features] # Extract features
        self.data_store.add_production_batch(features_df)
        print(f"[{self.name}] Ingested new data batch. Current history size: {len(self.data_store.production_data_history)}")

    def _run(self):
        # This agent is primarily event-driven by `process_incoming_data`, 
        # but could periodically clean up or log if needed.
        while self.running:
            time.sleep(1) # Just keep the thread alive

class DriftDetectionAgent(Agent):
    def __init__(self, name: str, message_bus: MessageBus, model_config: ModelConfig, data_store: DataStore):
        super().__init__(name, message_bus)
        self.model_config = model_config
        self.data_store = data_store
        self.drift_detectors: Dict[str, BaseDriftDetector] = {}
        # Simplified: one detector per feature for KSDrift
        for feature in model_config.features:
            self.drift_detectors[feature] = KSDrift(
                x_ref=self.data_store.baseline_data[[feature]].values, 
                p_val=0.05,
                alternative='two-sided',
                preprocess_X_ref=None, # Already preprocessed
                preprocess_X=None # Already preprocessed
            )
        print(f"[{self.name}] Initialized KSDrift detectors for features: {model_config.features}")

    def _run(self):
        self.message_bus.subscribe("monitor_alert", self._handle_monitor_alert)
        while self.running:
            time.sleep(1) # Wait for messages

    def _handle_monitor_alert(self, message: Dict[str, Any]):
        # In a real scenario, MonitorAgent would gather data, and this agent would be triggered.
        # For simplicity, we'll just check the latest data when triggered.
        print(f"[{self.name}] Received monitor alert. Checking for drift...")
        current_data = self.data_store.get_recent_production_data(num_batches=self.data_store.production_data_history.maxlen)
        
        if current_data.empty or self.data_store.baseline_data is None:
            print(f"[{self.name}] No current data or baseline available for drift detection.")
            return

        drifted_features = []
        overall_drift_score = 0.0
        detection_details = {}

        for feature in self.model_config.features:
            if feature not in current_data.columns:
                print(f"[{self.name}] Warning: Feature '{feature}' not found in current data. Skipping drift detection for this feature.")
                continue
            
            detector = self.drift_detectors.get(feature)
            if detector is None: # Handle cases where a detector wasn't initialized
                print(f"[{self.name}] No drift detector found for feature: {feature}. Skipping.")
                continue

            # Alibi-detect expects numpy arrays
            current_feature_data = current_data[[feature]].values
            
            # Ensure current data has enough samples for detection
            if len(current_feature_data) < detector.X_ref.shape[0]: # Using X_ref size as a heuristic for min_samples
                print(f"[{self.name}] Not enough samples ({len(current_feature_data)}) for feature '{feature}' drift detection yet. Skipping.")
                continue

            try:
                prediction = detector.predict(current_feature_data, return_p_val=True, return_distance=True)
                is_drift = prediction['data']['is_drift']
                p_val = prediction['data']['p_val'][0] if len(prediction['data']['p_val']) > 0 else 1.0
                distance = prediction['data']['distance'][0] if len(prediction['data']['distance']) > 0 else 0.0

                if is_drift:
                    drifted_features.append(feature)
                    overall_drift_score += (1 - p_val) # Inverse p-value as a simple score
                    detection_details[feature] = {"p_val": p_val, "distance": distance, "is_drift": bool(is_drift)}
                    print(f"[{self.name}] Drift detected for feature '{feature}' (p_val={p_val:.4f}, distance={distance:.4f})")
                else:
                    print(f"[{self.name}] No drift for feature '{feature}' (p_val={p_val:.4f})")
            except Exception as e:
                print(f"[{self.name}] Error during drift detection for feature '{feature}': {e}")

        if drifted_features:
            severity = 'high' if len(drifted_features) > len(self.model_config.features) / 2 else 'medium'
            drift_alert = DriftAlert(
                model_id=self.model_config.model_id,
                drift_type='data_drift',
                drifted_features=drifted_features,
                drift_score=overall_drift_score / len(drifted_features) if drifted_features else 0,
                severity=severity,
                details=detection_details
            )
            print(f"[{self.name}] --- Major Drift Alert! Publishing to diagnosis ---")
            self.message_bus.publish("drift_alert", drift_alert)
        else:
            print(f"[{self.name}] No significant drift detected across features.")


class MonitoringAgent(Agent):
    def __init__(self, name: str, message_bus: MessageBus, data_store: DataStore, data_config: DataConfig):
        super().__init__(name, message_bus)
        self.data_store = data_store
        self.data_config = data_config

    def _run(self):
        while self.running:
            if len(self.data_store.production_data_history) >= self.data_config.batch_size: # Trigger when enough new data
                print(f"[{self.name}] Monitoring interval reached. Triggering drift detection.")
                self.message_bus.publish("monitor_alert", {"status": "check_drift", "timestamp": time.time()})
            else:
                print(f"[{self.name}] Not enough data for monitoring yet. Current: {len(self.data_store.production_data_history)} / {self.data_config.batch_size}")
            time.sleep(self.data_config.monitoring_interval_seconds)


class DiagnosisAgent(Agent):
    def __init__(self, name: str, message_bus: MessageBus, model_config: ModelConfig):
        super().__init__(name, message_bus)
        self.model_config = model_config

    def _run(self):
        self.message_bus.subscribe("drift_alert", self._handle_drift_alert)
        while self.running:
            time.sleep(1)

    def _handle_drift_alert(self, alert: DriftAlert):
        print(f"[{self.name}] Received Drift Alert {alert.alert_id}. Diagnosing root cause...")
        diagnosis_report = self._perform_diagnosis(alert)
        remediation_plan = RemediationPlan(
            alert_id=alert.alert_id,
            model_id=alert.model_id,
            action='retrain' if alert.severity in ['medium', 'high'] else 'notify_human',
            diagnosis=diagnosis_report,
            priority='critical' if alert.severity == 'high' else 'high'
        )
        print(f"[{self.name}] Diagnosis complete. Publishing Remediation Plan {remediation_plan.plan_id}.")
        self.message_bus.publish("remediation_plan", remediation_plan)

    def _perform_diagnosis(self, alert: DriftAlert) -> str:
        """
        Placeholder for a sophisticated diagnosis logic. 
        In a real system, this would involve:
        - Analyzing feature distributions over time around the drift event.
        - Checking correlations between drifted features and target/other features.
        - Potentially using feature importance from the model to prioritize critical drifted features.
        - Comparing data statistics with external sources if available.
        """
        if alert.drifted_features:
            return f"Significant data drift detected in features: {', '.join(alert.drifted_features)}. This suggests changes in the input data distribution. Recommended action: model retraining with recent data to adapt to new patterns."
        return "Unspecified drift type or no specific features identified. Further manual investigation may be needed."


class RemediationAgent(Agent):
    def __init__(self, name: str, message_bus: MessageBus, retrain_callback: Callable[[DriftAlert], None]):
        super().__init__(name, message_bus)
        self.retrain_callback = retrain_callback
        # In a real system, this would interact with an MLOps platform like MLflow, Kubeflow, etc.

    def _run(self):
        self.message_bus.subscribe("remediation_plan", self._handle_remediation_plan)
        while self.running:
            time.sleep(1)

    def _handle_remediation_plan(self, plan: RemediationPlan):
        print(f"[{self.name}] Received Remediation Plan {plan.plan_id}. Action: {plan.action}")
        if plan.action == 'retrain':
            print(f"[{self.name}] Executing automated retraining for model {plan.model_id}...")
            # In a full system, we'd pass the actual DriftAlert or specific data context
            # For this example, we'll just pass the plan for context
            self.retrain_callback(plan) 
            print(f"[{self.name}] Retraining triggered for model {plan.model_id}.")
        elif plan.action == 'notify_human':
            print(f"[{self.name}] Notifying MLOps engineer for model {plan.model_id}. Diagnosis: {plan.diagnosis}")
            # Placeholder for sending email, slack notification, etc.
        else:
            print(f"[{self.name}] Unknown remediation action: {plan.action}. Doing nothing.")


# --- 4. Orchestrator ---

class DriftGuardOrchestrator:
    def __init__(self,
                 model_config: ModelConfig,
                 data_config: DataConfig,
                 baseline_data: pd.DataFrame,
                 prediction_function: Callable[[pd.DataFrame], np.ndarray], # Model's predict method
                 retrain_callback: Callable[[RemediationPlan], None]):
        
        self.model_config = model_config
        self.data_config = data_config
        self.prediction_function = prediction_function
        self.retrain_callback = retrain_callback

        self.message_bus = MessageBus()
        self.data_store = DataStore(history_size=data_config.history_size)
        self.data_store.set_baseline(baseline_data[model_config.features])

        # Initialize agents
        self.data_ingestion_agent = DataIngestionAgent("DataIngestionAgent", self.message_bus, self.data_store, self.model_config)
        self.monitoring_agent = MonitoringAgent("MonitoringAgent", self.message_bus, self.data_store, self.data_config)
        self.drift_detection_agent = DriftDetectionAgent("DriftDetectionAgent", self.message_bus, self.model_config, self.data_store)
        self.diagnosis_agent = DiagnosisAgent("DiagnosisAgent", self.message_bus, self.model_config)
        self.remediation_agent = RemediationAgent("RemediationAgent", self.message_bus, self.retrain_callback)

        self.agents = [
            self.data_ingestion_agent,
            self.monitoring_agent,
            self.drift_detection_agent,
            self.diagnosis_agent,
            self.remediation_agent
        ]
        
        self.message_bus_thread = threading.Thread(target=self.message_bus._process_messages, name="MessageBusThread", daemon=True)

    def start(self):
        print("Starting DriftGuard Orchestrator and agents...")
        self.message_bus_thread.start()
        for agent in self.agents:
            agent.start()
        print("DriftGuard Orchestrator is operational.")

    def process_data_batch(self, df: pd.DataFrame):
        """ External API for feeding new production data. """
        # Add model predictions to the data if target_column is not present but needed for concept drift
        # (Not implemented in this KSDrift focused example, but important for full functionality)
        # if self.model_config.target_column is None and 'prediction' not in df.columns:
        #    df['prediction'] = self.prediction_function(df[self.model_config.features])
        
        self.data_ingestion_agent.process_incoming_data(df)

    def shutdown(self):
        print("Shutting down DriftGuard Orchestrator and agents...")
        for agent in reversed(self.agents):
            agent.stop()
        # The message bus thread is daemon, will exit with main program
        print("DriftGuard Orchestrator shut down.")


# --- Example Placeholder Functions (for demonstration) ---

def _mock_predict(df: pd.DataFrame) -> np.ndarray:
    """ A simple mock prediction function for demonstration. """
    # Sum features and add some noise
    return (df.sum(axis=1) + np.random.rand(len(df)) * 0.5).values

def _mock_retrain_pipeline(plan: RemediationPlan):
    """ A mock function representing an MLOps retraining pipeline trigger. """
    print(f"[MOCK RETRAIN PIPELINE] Received remediation plan for model {plan.model_id}:")
    print(f"  Diagnosis: {plan.diagnosis}")
    print(f"  Timestamp: {plan.timestamp}")
    # In a real system, this would call an external API or trigger a specific job.
    print("  -> Initiating new model training job with latest data...")
    time.sleep(2) # Simulate pipeline run time
    print("  -> Model training job completed successfully.")

