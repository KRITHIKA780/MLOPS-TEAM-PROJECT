"""Airflow DAG: ingest -> validate -> features -> train -> evaluate -> quality_gate -> register -> deploy.

Runs daily, or can be triggered by a drift alert (airflow dags trigger recsys_training_pipeline).
The project root must be on PYTHONPATH (docker-compose sets PYTHONPATH=/opt/project).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

from src import pipeline
from src.utils import load_config

args = {"owner": "group6", "retries": 2, "retry_delay": timedelta(minutes=5)}


def _run(stage):
    def fn():
        return pipeline.STAGES[stage](load_config())
    return fn


def better_than_prod():
    return "register_model" if pipeline.gate(load_config()) else "keep_current_model"


with DAG("recsys_training_pipeline", start_date=datetime(2026, 1, 1), schedule="@daily", catchup=False,
         default_args=args, tags=["recsys", "mlops"]) as dag:
    ingest = PythonOperator(task_id="ingest_data", python_callable=_run("ingest"))
    validate = PythonOperator(task_id="validate_data", python_callable=_run("validate"))
    features = PythonOperator(task_id="feature_engineering", python_callable=_run("features"))
    train = PythonOperator(task_id="train_models", python_callable=_run("train"))
    evaluate = PythonOperator(task_id="evaluate_models", python_callable=_run("evaluate"))
    gate = BranchPythonOperator(task_id="quality_gate", python_callable=better_than_prod)
    register = PythonOperator(task_id="register_model", python_callable=_run("register"))
    deploy = PythonOperator(task_id="deploy_model", python_callable=_run("deploy"))
    skip = PythonOperator(task_id="keep_current_model", python_callable=lambda: None)

    ingest >> validate >> features >> train >> evaluate >> gate >> [register, skip]
    register >> deploy
