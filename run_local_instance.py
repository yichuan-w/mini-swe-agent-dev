#!/usr/bin/env python3
"""Run swebench-single on a local modified dataset instance."""

import json
import sys
import traceback
from pathlib import Path

import typer
import yaml

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import get_sb_environment, update_preds_file
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT = global_config_dir / "last_swebench_single_run.traj.json"


@app.command()
def main(
    json_file: Path = typer.Argument(..., help="Path to local JSON dataset file"),
    instance_id: str = typer.Option(None, "-i", "--instance", help="Instance ID (if not provided, will list all instances)"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use"),
    config_path: Path = typer.Option(builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file"),
    environment_class: str | None = typer.Option(None, "--environment-class"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
) -> None:
    """Run on a single instance from a local JSON dataset file."""
    
    # Load local JSON dataset
    logger.info(f"Loading dataset from {json_file}...")
    with open(json_file) as f:
        instances = json.load(f)
    
    # Create instance dict
    instances_dict = {inst["instance_id"]: inst for inst in instances}
    
    # If no instance_id provided, list all instances
    if instance_id is None:
        logger.info(f"Found {len(instances_dict)} instances:")
        for i, inst_id in enumerate(sorted(instances_dict.keys())):
            print(f"  {i}: {inst_id}")
        print(f"\nUsage: python run_local_instance.py {json_file} -i <instance_id> -m <model>")
        return
    
    # Get the instance
    if instance_id not in instances_dict:
        logger.error(f"Instance {instance_id} not found in dataset!")
        logger.info(f"Available instances: {', '.join(sorted(instances_dict.keys())[:10])}...")
        return
    
    instance = instances_dict[instance_id]
    logger.info(f"Running on instance: {instance_id}")
    
    # Load config
    config_path = get_config_path(config_path)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    
    # Create environment and model
    env = get_sb_environment(config, instance)
    model = get_model(model_name, config.get("model", {}))
    agent = InteractiveAgent(
        model,
        env,
        **({"mode": "yolo"} | config.get("agent", {})),
    )
    
    # Run the agent
    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(instance["problem_statement"])
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)
        # Generate preds.json file for submission
        preds_path = output.parent / "preds.json" if output.is_file() else output / "preds.json"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        update_preds_file(preds_path, instance["instance_id"], model.config.model_name, result or "")


if __name__ == "__main__":
    app()

