"""
nni_encoding_param_optimization.py

Creates and starts the NNI experiment.

NNI launches nni_encoding_param_optimization_backbone.py once
for every sampled encoding configuration.

If nni hangs run the following:
sudo lsof -nP -iTCP:8080 -sTCP:LISTEN
example output:
COMMAND    PID         USER FD   TYPE  DEVICE SIZE/OFF NODE NAME
node    355901 smullercleve 21u  IPv6 7326098      0t0  TCP *:8080 (LISTEN)

now run:
kill <PID>
"""

import sys

from nni import experiment
from nni.experiment import Experiment

search_space = {
    "threshold": {
        "_type": "choice",
        "_value": [1, 2, 3, 4, 5],
    },
    "max_frequency": {
        "_type": "randint",
        "_value": [10, 101],  # 10–100; upper bound is exclusive
    },
    "refractory_period_ms": {
        "_type": "randint",
        "_value": [0, 11],  # 0–10
    },
}


def main() -> None:
    experiment = Experiment("local")

    # Explicitly invoke (virtualenv) Python.
    experiment.config.trial_command = (
        f'"{sys.executable}" '
        "-m utils.nni_encoding_param_optimization_backbone"
    )
    experiment.config.trial_code_directory = "."

    experiment.config.search_space = search_space

    experiment.config.tuner.name = "anneal"
    experiment.config.tuner.class_args = {
        "optimize_mode": "maximize",
    }

    # experiment.config.max_trial_number = 1000
    # experiment.config.trial_concurrency = 5
    # experiment.config.max_experiment_duration = "168h"
    
    # for a quick test
    parallel_trials = 2
    experiment.config.max_trial_number = 1000
    experiment.config.trial_concurrency = parallel_trials
    experiment.config.max_experiment_duration = "168h"  # 7 days

    # Each trial requests one GPU, with up to five trials sharing that GPU.
    experiment.config.trial_gpu_number = 1
    experiment.config.training_service.use_active_gpu = True
    experiment.config.training_service.max_trial_number_per_gpu = parallel_trials

    experiment.config.debug = False
    experiment.config.log_level = "info"

    experiment.run(
        port=8080,
        debug=False,
    )


if __name__ == "__main__":
    main()
