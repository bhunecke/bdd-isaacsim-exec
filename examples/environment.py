import sys
import os
import time
import json
import yaml
from json import JSONDecodeError
from behave.model import Feature, Scenario, Step
from rdflib import ConjunctiveGraph
from behave.runner import Context
from rdf_utils.uri import URL_SECORO_M
from rdf_utils.resolver import install_resolver
from rdf_utils.naming import get_valid_var_name
from bdd_isaacsim_exec.behave import before_all_isaac, before_scenario_isaac, after_scenario_isaac


DEFAULT_ISAAC_PHYSICS_DT_SEC = 1.0 / 60.0


def before_all(context: Context):
    from pprint import pprint

    # config file
    config_file = context.config.userdata.get("config_file", None)
    assert config_file is not None, "no config file specified"
    assert os.path.isfile(config_file), f"not a file: {config_file}"
    with open(config_file, "r") as cf:
        configs = yaml.safe_load(cf)

    if "hide_ui" not in configs:
        configs["hide_ui"] = False
    if "time_step_sec" not in configs:
        configs["time_step_sec"] = DEFAULT_ISAAC_PHYSICS_DT_SEC
    print("Configurations:")
    pprint(configs)

    # model file
    model_file = context.config.userdata.get("model_file", None)
    assert model_file is not None, "no model file specified"
    assert os.path.isfile(model_file), f"not a file: {model_file}"
    with open(model_file, "r") as mf:
        models = yaml.safe_load(mf)

    install_resolver()
    g = ConjunctiveGraph()
    for model_data in models:
        assert "url" in model_data and "format" in model_data, f"invalid model info: {model_data}"
        url = model_data["url"]
        fmt = model_data["format"]
        try:
            g.parse(url, format=fmt)
        except JSONDecodeError as e:
            print(f"error parsing '{url}' into graph (format='{fmt}'):\n{e}")
            sys.exit(1)

    context.model_graph = g
    context.exec_timestamp = time.strftime("%Y%m%d-%H%M%S")
    before_all_isaac(context=context, sim_configs=configs)


def before_feature(context: Context, feature: Feature):
    context.log_data = {}
    if context.enable_capture:
        assert (
            context.capture_configs is not None and "root_dir" in context.capture_configs
        ), f"invalid capture configs: {context.capture_configs}"
        context.capture_folder = os.path.join(
            context.capture_configs["root_dir"],
            f"capture-{get_valid_var_name(feature.name)}-{context.exec_timestamp}",
        )
        print("Creating directory: ", context.capture_folder)
        os.makedirs(name=context.capture_folder, exist_ok=True)


def after_feature(context: Context, feature: Feature):
    log_data_file = os.path.join(
        context.capture_folder,
        f"log_data-{get_valid_var_name(feature.name)}-{context.exec_timestamp}.json",
    )
    with open(log_data_file, "w") as file:
        file.write(json.dumps(context.log_data, indent=2))


def before_scenario(context: Context, scenario: Scenario):
    context.log_data[scenario.name] = {"clauses": [], "cameras": []}
    context.log_data[scenario.name]["start_time_unix"] = time.time()
    context.frame_logs = []
    before_scenario_isaac(context, scenario)


def after_scenario(context: Context, scenario: Scenario):
    end_time = time.time()
    context.log_data[scenario.name]["end_time_unix"] = end_time
    context.log_data[scenario.name]["exec_time"] = (
        end_time - context.log_data[scenario.name]["start_time_unix"]
    )
    context.log_data[scenario.name]["behave_exec_time"] = scenario.duration

    if context.enable_capture:
        frame_data_file = os.path.join(
            context.scenario_capture_folder,
            f"frame_data-{get_valid_var_name(scenario.name)}-{context.exec_timestamp}.json",
        )
        os.makedirs(os.path.dirname(frame_data_file), exist_ok=True)
        with open(frame_data_file, "w") as file:
            file.write(json.dumps(context.frame_logs, indent=2))
    after_scenario_isaac(context)


def before_step(context: Context, step: Step):
    context.step_debug_info = {"name": step.name, "keyword": step.keyword, "fail_info": {}}


def after_step(context: Context, step: Step):
    context.step_debug_info["exec_time"] = step.duration
    context.step_debug_info["status"] = step.status.name
    context.log_data[context.scenario.name]["clauses"].append(context.step_debug_info)
