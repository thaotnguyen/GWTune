# %%
import functools
import os
import warnings
from typing import Any, Dict, Optional, List

import numpy as np
import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState
from sqlalchemy_utils import create_database, database_exists


class _NoImprovementEarlyStoppingCallback:
    """Stop the study when the best objective fails to improve for ``patience`` completed trials."""

    def __init__(self, patience: int, min_completed_trials: int) -> None:
        self._patience = max(1, patience)
        self._min_completed_trials = max(1, min_completed_trials)
        self._best_value: Optional[float] = None
        self._no_improve_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        if n_complete < self._min_completed_trials:
            return
        best = study.best_value
        if best is None:
            return
        if self._best_value is None:
            self._best_value = best
            return
        if study.direction == StudyDirection.MAXIMIZE:
            improved = best > self._best_value + 1e-12
        else:
            improved = best < self._best_value - 1e-12
        if improved:
            self._best_value = best
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
            if self._no_improve_count >= self._patience:
                study.stop()


# %%
class RunOptuna:
    """A class to manage and run optimization studies using Optuna.

    This class provides easy configuration and management for optimization studies using Optuna, offering
    methods for selecting samplers and pruners, defining the study's search space, and running the study
    across multiple trials. This class Also includes utility functions for epsilon space definition, sampler
    and pruner selection, and study loading/creation.

    Attributes:
        filename (str): The filename to save the results.
        storage (str): Storage URL for the database. For SQLite, this is a file path.
        init_mat_plan (str): The method to be used for the initial plan.
        num_trial (int): Number of trials for the optimization.
        n_iter (int): Number of initial plans evaluated during a single optimization.
        n_jobs (int): Number of jobs to run in parallel.
        sampler_name (str): Name of the sampler used in optimization. Options are "random", "grid", and "tpe".
        pruner_name (str): Name of the pruner used in optimization. Options are "hyperband", "median", and "nop".
        pruner_params (dict, optional): Additional parameters for the pruner. See Optuna's pruner page for more details.

        n_startup_trials (int): Number of trials to be evaluated before using the pruner (specifically for MedianPruner).
        n_warmup_steps (int): Number of warm-up steps before using the pruner (specifically for MedianPruner).
        min_resource (int): Minimum resource allocated for a trial (specifically for HyperbandPruner).
        reduction_factor (int): Division factor for the allocated resource of a trial (specifically for HyperbandPruner).
        search_space (dict, optional): Search space definition for grid search sampler.
    """
    def __init__(
        self,
        filename: str,
        storage: str,
        init_mat_plan: str,
        num_trial: int,
        n_iter: int,
        n_jobs: int,
        sampler_name: str,
        pruner_name: str,
        pruner_params: Optional[dict] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_trials: int = 15,
        max_total_trials: Optional[int] = None,
    ) -> None:
        """Initialize the RunOptuna class which serves as a utility for handling and running Optuna studies.

        Args:
            filename (str): The filename to save the results.
            storage (str): Storage URL for the database. For SQLite, this is a file path.
            init_mat_plan (str): The method to be used for the initial plan. Options are
                                 "uniform", "diag", "random", "permutation" or "user_define".
            num_trial (int):  Number of trials for optimization.
            n_iter (int): Number of initial plans evaluated during a single optimization.
            n_jobs (int): Number of jobs to run in parallel.
            sampler_name (str): Name of the sampler used in optimization. Options are "random", "grid", and "tpe".
            pruner_name (str):  Name of the pruner used in optimization. Options are "hyperband", "median", and "nop".
            pruner_params (Optional[dict], optional): Additional parameters for the pruner. See Optuna's pruner page for more details. Defaults to None.
            early_stopping_patience (Optional[int], optional): If set, stop the study after this many
                consecutive completed trials without improving the best value. Defaults to None (disabled).
            early_stopping_min_trials (int, optional): Minimum number of completed trials before early
                stopping is allowed. Defaults to 15.
            max_total_trials (Optional[int], optional): If set, cap the total number of persisted Optuna
                trial rows for the study, including completed, pruned, failed, and running trials.
        """

        # the path or file name to save the results.
        self.filename = filename
        self.storage = storage
        self.init_mat_plan = init_mat_plan

        # parameters for optuna.study
        self.num_trial = num_trial
        self.n_iter = n_iter
        self.n_jobs = n_jobs

        # setting of optuna
        self.sampler_name = sampler_name
        self.pruner_name = pruner_name
        self.pruner_params = pruner_params
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_trials = early_stopping_min_trials
        self.max_total_trials = max_total_trials

        # parameters for optuna.study
        self.n_jobs = n_jobs
        self.num_trial = num_trial


        # parameters for optuna.study
        self.n_jobs = n_jobs
        self.num_trial = num_trial

        # MedianPruner
        self.n_startup_trials = 5
        self.n_warmup_steps = 5

        # HyperbandPruner
        self.min_resource = 5
        self.reduction_factor = 2

        if pruner_params is not None:
            self._set_params(pruner_params)

    def _set_params(self, vars_dic: dict) -> None:
        """Set the attributes of the class based on the provided dictionary.

        Args:
            vars_dic (dict): Dictionary containing attribute names as keys and the desired values.

        Raises:
            Warning: A warning is raised if a key in the dictionary does not correspond to an attribute of the class.
        """
        for key, value in vars_dic.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a parameter of the pruner.")

    def create_study(self, direction: str = "minimize", seed: int = 42) -> optuna.study.Study:
        """Create a new Optuna study.

        This method initializes a new study for the optimization process.
        If a study with the same name already exists, it will be loaded instead of creating a new one.

        Args:
            direction (str, optional): Optimization direction for the study. Either "minimize" for minimization problems or
                                       "maximize" for maximization problems. Defaults to "minimize".
            seed (int, optional): Seed for the sampler when creating a new study. Defaults to 42.

        Returns:
            optuna.study.Study: Initialized Optuna study.
        """
        study = optuna.create_study(
            direction=direction,
            study_name=self.filename + "_" + self.init_mat_plan,
            storage=self.storage,
            load_if_exists=True,
            sampler=self.choose_sampler(seed=seed, constant_liar=(self.n_jobs > 1)),
            pruner=self.choose_pruner(),
        )
        return study

    def load_study(self, seed: int = 42, compute_OT=True) -> optuna.study.Study:
        """Load an existing Optuna study from the database.

        If a study with the specified name exists in the database, it will be loaded.
        This method uses the specified sampler and pruner for the loaded study.

        Args:
            seed (int, optional): Seed for random number generation. Useful for reproducibility. Defaults to 42.

        Returns:
            optuna.study.Study: Loaded Optuna study.
        """
        
        if compute_OT:
            study = optuna.load_study(
                study_name=self.filename + "_" + self.init_mat_plan,
                sampler=self.choose_sampler(seed=seed, constant_liar=(self.n_jobs > 1)),
                pruner=self.choose_pruner(),
                storage=self.storage,
            )
        else:
            study = optuna.load_study(
                study_name=self.filename + "_" + self.init_mat_plan,
                storage=self.storage,
            )
        return study

    def run_study(
        self,
        objective: Any,
        seed: int = 42,
        **kwargs
    ) -> optuna.study.Study:
        """Execute the Optuna study to optimize the provided objective function.

        The study can be executed in parallel using the specified number of jobs. Depending on the chosen sampler, a search space can be specified for grid search.

        Args:
            objective (Any): The objective function to be optimized.
            seed (int, optional): Seed for random number generation. Useful for reproducibility. Defaults to 42.
            **kwargs: Additional keyword arguments. Can include "search_space" for grid search.

        Returns:
            optuna.study.Study: An Optuna study object containing the results of the optimization.

        Raises:
            ValueError: If a necessary parameter is missing.
            UserWarning: If a search space is provided but not used by the chosen sampler.
        """
        if self.sampler_name == "grid":
            assert kwargs.get("search_space") != None, "please define search space for grid search."
            self.search_space = kwargs.pop("search_space")

        else:
            if kwargs.get("search_space") is not None:
                warnings.warn("except for grid search, search space is ignored.", UserWarning)
            del kwargs["search_space"]

        try:
            # If there is no .db file or database of MySQL, multi_run will not work properly if you don't let it load here.
            study = self.load_study(seed=seed)
        except KeyError:
            print("Study for " + self.filename + "_" + self.init_mat_plan + " was not found, creating a new one...")
            self.create_study(seed=seed)
            study = self.load_study(seed=seed)

        objective = functools.partial(objective, **kwargs)

        n_trials = self.num_trial
        if self.max_total_trials is not None:
            remaining_trials = max(0, self.max_total_trials - len(study.trials))
            if remaining_trials == 0:
                print(
                    f"Study {self.filename}_{self.init_mat_plan} already has "
                    f"{len(study.trials)} trials; skipping because max_total_trials="
                    f"{self.max_total_trials}."
                )
                return study
            n_trials = min(n_trials, remaining_trials)

        if self.n_jobs > 1:
            warnings.filterwarnings("always")
            warnings.warn(
                "The parallel computation is done by the functions implemented in Optuna.\n \
                This doesn't always provide a benefit to speed up or to get a better results.",
                UserWarning,
            )
            warnings.filterwarnings("ignore")

        callbacks: List[Any] = []
        if self.early_stopping_patience is not None and self.early_stopping_patience > 0:
            callbacks.append(
                _NoImprovementEarlyStoppingCallback(
                    patience=self.early_stopping_patience,
                    min_completed_trials=self.early_stopping_min_trials,
                )
            )

        study.optimize(
            objective,
            n_trials,
            n_jobs=self.n_jobs,
            callbacks=callbacks or None,
        )

        return study

    def choose_sampler(self, seed: int = 42, constant_liar: bool = False, multivariate: bool = False) -> optuna.samplers.BaseSampler:
        """Choose and instantiate an Optuna sampler based on the specified sampler name.

        This method supports a number of predefined sampler names, such as "random", "grid", and "tpe".
        For the TPE sampler, additional parameters can be set, such as "constant_liar" and "multivariate".

        Args:
            seed (int, optional): Seed for random number generation in the sampler. Useful for reproducibility. Defaults to 42.
            constant_liar (bool, optional): For TPE sampler, whether to use constant liar. It can be beneficial for distributed optimization. Defaults to False.
            multivariate (bool, optional): For TPE sampler, whether to use multivariate TPE. Defaults to False.

        Returns:
            optuna.samplers.BaseSampler: An instantiated sampler object.

        Raises:
            ValueError: If an unsupported sampler name is provided.

        Note:
            For more details, please refer to the Optuna sampler documentation.
        """
        if self.sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed)

        elif self.sampler_name == "grid":
            sampler = optuna.samplers.GridSampler(self.search_space, seed=seed)

        elif self.sampler_name.lower() == "tpe":
            sampler = optuna.samplers.TPESampler(
                constant_liar=constant_liar,  # I heard it is better to set to True for distributed optimization (Abe)
                multivariate=multivariate,
                seed=seed,
            )

        else:
            raise ValueError("not implemented sampler yet.")

        return sampler

    def choose_pruner(self) -> optuna.pruners.BasePruner:
        """Choose the appropriate pruner based on the specified pruner name.

        For pruner selection, you can choose from "median", "hyperband", or "nop". It's often recommended to use the RandomSampler
        with the MedianPruner or the TPESampler with the HyperbandPruner for optimal results.

        Returns:
            optuna.pruners.BasePruner: An instance of the chosen pruner.

        Raises:
            ValueError: If the pruner name is not recognized.

        Note:
            For more details, please refer to the Optuna pruner documentation.
        """
        if self.pruner_name == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps
            )
        elif self.pruner_name.lower() == "hyperband":
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=self.min_resource,
                max_resource=self.n_iter,
                reduction_factor=self.reduction_factor
            )
        elif self.pruner_name.lower() == "nop":
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError("not implemented pruner yet.")

        return pruner


def load_optimizer(
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    storage: Optional[str] = None,
    init_mat_plan: str = "random",
    n_iter: int = 10,
    num_trial: int = 20,
    n_jobs: int = 1,
    method: str = "optuna",
    sampler_name: str = "random",
    pruner_name: str = "median",
    pruner_params: Optional[Dict[str, Any]] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_trials: int = 15,
    max_total_trials: Optional[int] = None,
) -> RunOptuna:
    """Loads and initializes the optimizer for hyperparameter tuning.

    Usage example:
    >>> dataset = mydataset()
    >>> opt = load_optimizer(filename)
    >>> study = Opt.run_study(dataset)

    Args:
        save_path (str, optional): Directory where the results of optimization will be saved.
                                   If it doesn't exist, a new directory will be created.
        filename (str, optional): Name of the file where optimizer data is stored.
        storage (str, optional): URL to the database storage.
        init_mat_plan (str, optional): The method to be used for the initial plan. Options are "uniform",
                                       "diag", "random", "permutation" or "user_define".
        n_iter (int, optional): Number of initial plans evaluated during a single optimization. Defaults to 10.
        num_trial (int, optional): Number of trials for optimization. Defaults to 20.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
        method (str, optional): Optimization method. Currently, only "optuna" is supported. Defaults to "optuna".
        sampler_name (str, optional): Name of the sampler used in optimization. Options are "random", "grid", and "tpe". Defaults to "random".
        pruner_name (str, optional): Name of the pruner used in optimization. Options are "hyperband", "median", and "nop". Defaults to "median".
        pruner_params (dict, optional): Additional parameters for the pruner. See Optuna's pruner page for more details
        early_stopping_patience (int, optional): Stop after this many completed trials without improving the best value.
        early_stopping_min_trials (int, optional): Minimum completed trials before early stopping applies.
        max_total_trials (int, optional): Cap total persisted Optuna trial rows for the study.

    Returns:
        opt : instance of optimzer.
    """

    # make file_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # create a database from the URL
    if not database_exists(storage):
        create_database(storage)

    if method == "optuna":
        Opt = RunOptuna(
            filename,
            storage,
            init_mat_plan,
            num_trial,
            n_iter,
            n_jobs,
            sampler_name,
            pruner_name,
            pruner_params,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_trials=early_stopping_min_trials,
            max_total_trials=max_total_trials,
        )
    else:
        raise ValueError("no implemented method.")

    return Opt
