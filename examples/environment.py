import sys
import os
import time
import yaml
import json
from json import JSONDecodeError
from behave.model import Feature, Scenario, Step
from rdflib import ConjunctiveGraph
from behave.runner import Context
from rdf_utils.uri import URL_SECORO_M
from rdf_utils.resolver import install_resolver
from rdf_utils.naming import get_valid_var_name
from bdd_isaacsim_exec.behave import before_all_isaac, before_scenario_isaac, after_scenario_isaac


LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(name=LOG_DIR, exist_ok=True)


MODELS = {
    f"{URL_SECORO_M}/acceptance-criteria/bdd/agents/isaac-sim.agn.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/scenes/isaac-agents.scene.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/environments/secorolab.env.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/simulation/secorolab-isaac.sim.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/scenes/secorolab-env.scene.json": "json-ld",
    # f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/pickplace.tmpl.json": "json-ld",
    # f"{URL_SECORO_M}/acceptance-criteria/bdd/pickplace-secorolab-isaac.var.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/sorting.tmpl.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/sorting-secorolab-isaac.var.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/execution/pickplace-secorolab-isaac.exec.json": "json-ld",
    f"{URL_SECORO_M}/acceptance-criteria/bdd/execution/pickplace-secorolab-isaac.bhv.exec.json": "json-ld",
}

DEFAULT_ISAAC_PHYSICS_DT_SEC = 1.0 / 60.0


def before_all(context: Context):
    install_resolver()
    read_config_file(context)
    g = ConjunctiveGraph()
    for url, fmt in MODELS.items():
        try:
            g.parse(url, format=fmt)
        except JSONDecodeError as e:
            print(f"error parsing '{url}' into graph (format='{fmt}'):\n{e}")
            sys.exit(1)

    context.model_graph = g
    if context.use_livestream:
        assert context.render is True, "Livestream mode requires rendering. Set 'render' to True."
        assert context.headless is True, "Livestream mode requires headless mode. Set 'headless' to True."
    before_all_isaac(context=context, headless=context.headless, time_step_sec=DEFAULT_ISAAC_PHYSICS_DT_SEC)

def read_config_file(context, filename="config.yaml"):
    config_path = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(config_path):
        print(f"Config file '{config_path}' does not exist.")
        return

    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            if config and isinstance(config, dict):
                for key, value in config.items():
                    setattr(context, key, value)
        except yaml.YAMLError as e:
            print(f"Error reading config file '{config_path}': {e}")

def before_feature(context: Context, feature: Feature):
    context.log_data = {}


def after_feature(context: Context, feature: Feature):
    log_data_file = os.path.join(
        LOG_DIR,
        f"log_data-{get_valid_var_name(feature.name)}-{time.strftime('%Y%m%d-%H%M%S')}.json",
    )
    with open(log_data_file, "w") as file:
        file.write(json.dumps(context.log_data))


def before_scenario(context: Context, scenario: Scenario):
    context.log_data[scenario.name] = {"clauses": []}
    context.scenario_start_time = time.process_time()
    before_scenario_isaac(context, scenario)


def after_scenario(context: Context, scenario: Scenario):
    scr_exec_time = time.process_time() - context.scenario_start_time
    context.log_data[scenario.name]["exec_time"] = scr_exec_time
    after_scenario_isaac(context)


def before_step(context: Context, step: Step):
    context.step_debug_info = {"name": step.name, "keyword": step.keyword, "fail_info": {}}
    step_start = time.process_time()
    context.step_start = step_start


def after_step(context: Context, step: Step):
    step_exec_time = time.process_time() - context.step_start
    context.step_debug_info["exec_time"] = step_exec_time
    context.step_debug_info["status"] = step.status.name
    context.log_data[context.scenario.name]["clauses"].append(context.step_debug_info)
