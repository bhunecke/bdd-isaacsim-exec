import sys
import os
import time
import json
from json import JSONDecodeError
from behave.model import Feature, Scenario, Step
from rdflib import ConjunctiveGraph
from behave.runner import Context
from rdf_utils.uri import URL_SECORO_M
from rdf_utils.resolver import install_resolver
from rdf_utils.naming import get_valid_var_name
from bdd_isaacsim_exec.behave import before_all_isaac, before_scenario_isaac, after_scenario_isaac


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
    g = ConjunctiveGraph()
    for url, fmt in MODELS.items():
        try:
            g.parse(url, format=fmt)
        except JSONDecodeError as e:
            print(f"error parsing '{url}' into graph (format='{fmt}'):\n{e}")
            sys.exit(1)

    context.model_graph = g
    context.exec_timestamp = time.strftime('%Y%m%d-%H%M%S')
    before_all_isaac(context=context, render_type="normal", enable_capture=True, time_step_sec=DEFAULT_ISAAC_PHYSICS_DT_SEC)


def before_feature(context: Context, feature: Feature):
    context.log_data = {}
    context.root_capture_folder = os.path.join(
        os.path.dirname(__file__),
        "captures",
        f"capture-{get_valid_var_name(feature.name)}-{context.exec_timestamp}"
    )
    os.makedirs(name=context.root_capture_folder, exist_ok=True)


def after_feature(context: Context, feature: Feature):
    log_data_file = os.path.join(
        context.root_capture_folder,
        f"log_data-{get_valid_var_name(feature.name)}-{context.exec_timestamp}.json",
    )
    with open(log_data_file, "w") as file:
        file.write(json.dumps(context.log_data))


def before_scenario(context: Context, scenario: Scenario):
    context.log_data[scenario.name] = {"clauses": [], "cameras": []}
    context.log_data[scenario.name]["start_time_unix"] = time.time()
    context.frame_logs = []
    before_scenario_isaac(context, scenario)


def after_scenario(context: Context, scenario: Scenario):
    end_time = time.time()
    context.log_data[scenario.name]["end_time_unix"] = end_time
    context.log_data[scenario.name]["exec_time"] = end_time - context.log_data[scenario.name]["start_time_unix"]
    context.log_data[scenario.name]["behave_exec_time"] = scenario.duration
    frame_data_file = os.path.join(
        context.scenario_capture_folder,
        f"frame_data-{get_valid_var_name(scenario.name)}-{context.exec_timestamp}.json",
    )
    with open(frame_data_file, "w") as file:
        file.write(json.dumps(context.frame_logs))
    after_scenario_isaac(context)


def before_step(context: Context, step: Step):
    context.step_debug_info = {"name": step.name, "keyword": step.keyword, "fail_info": {}}


def after_step(context: Context, step: Step):
    context.step_debug_info["exec_time"] = step.duration
    context.step_debug_info["status"] = step.status.name
    context.log_data[context.scenario.name]["clauses"].append(context.step_debug_info)
