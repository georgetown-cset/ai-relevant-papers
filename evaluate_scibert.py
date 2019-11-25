"""Run SciBERT models over arXiv for evaluation.

Hacked up from run_as_library.py.
"""
import apache_beam as beam
import argparse
import datetime
import json
import os
import pickle
import scibert
import allennlp

from apache_beam.options.pipeline_options import PipelineOptions
from subprocess import Popen, PIPE


class PredictDoFn(beam.DoFn):
    def __init__(self, model_name, remote_model_parent):
        self.model_name = model_name
        self.remote_model_parent = remote_model_parent
        self.predictor = None
        self.run_date = datetime.datetime.now().strftime("%Y-%m-%d")

    def prepare_model(self):
        model_prefix = "model-parent-"
        tmp_dir = os.path.join(os.getcwd(), model_prefix + str(os.getpid()))
        prev_model_dirs = [f for f in os.listdir(os.getcwd()) if f.startswith(model_prefix) and os.path.exists(
            os.path.join(os.getcwd(), f, self.model_name, "z_done"))]
        # attempt to grab an old model dir to avoid re-downloading the models
        if len(prev_model_dirs) > 0:
            model_dir = os.path.join(os.getcwd(), prev_model_dirs[0], self.model_name)
        else:
            # if we enter this block, then we weren't able to load an old model, so we need to download the model,
            # populate its config, then load it
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            for model in ["scibert_scivocab_uncased", self.model_name]:
               models_cmd = f"gsutil cp -r gs://{self.remote_model_parent}/{model} {tmp_dir}"
               proc = Popen(models_cmd, shell=True, stdout=PIPE, stderr=PIPE)
               output, _ = proc.communicate()
            model_dir = os.path.join(tmp_dir, self.model_name)
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            cfg = open(os.path.join(model_dir, "config-template.json")).read()
            cfg = cfg.replace("PATH_TO_MODELS", os.path.sep.join(model_dir.split(os.path.sep)[:-1]))
            open(os.path.join(model_dir, "config.json"), "w").write(cfg)
        return model_dir

    def start_bundle(self):
        if self.predictor is not None:
            return
        model_dir = self.prepare_model()
        # the following line is a necessary bad import practice, otherwise beam tries to serialize allennlp and the
        # deserialization breaks on dataflow.
        from scibert.models import text_classifier
        from scibert.predictors.predictor import ScibertPredictor
        from allennlp.predictors import Predictor
        import scibert
        self.predictor = Predictor.from_path(model_dir, predictor_name="text_classifier")

    def process(self, record):
        js = json.loads(record)
        js['sentence'] = js.pop('text')
        if len(js["sentence"].strip()) > 0:
            results = self.predictor.predict_json(js)
            results["id"] = js.get("meta")
            results["model_name"] = "scibert-" + self.model_name
            results["run_date"] = self.run_date
            yield results


def run_pipeline(input_path, model_names, model_dir, output_path):
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        input_data = p | "Read From GCS" >> beam.io.ReadFromText(input_path)
        for model_name in model_names:
            (input_data | "Run SciBERT with " + model_name >> beam.ParDo(PredictDoFn(model_name, model_dir))
             | "Write Out " + model_name + " predictions" >> beam.io.WriteToText(output_path + "-" + model_name,
                                                                                 file_name_suffix=".jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("model_list", help="list of models to run, separated with commas")
    parser.add_argument("output_path")
    parser.add_argument("--model_dir", help="gcs location of parent directory of models in model_list")
    args, pipeline_args = parser.parse_known_args()
    run_pipeline(args.input_path, args.model_list.split(","), args.model_dir, args.output_path)
