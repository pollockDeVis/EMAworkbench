import pickle
import sys

from ema_workbench import (SequentialEvaluator, MultiprocessingEvaluator, MPIEvaluator, perform_experiments)
from models_to_benchmark import get_python_model_instance, get_lake_model_instance, get_flu_model_instance
import timeit

def run_model_with_evaluator(evaluator_class, model_function, scenarios, **kwargs):
    model = model_function()
    with evaluator_class(model, **kwargs) as evaluator:  # Pass the **kwargs to the evaluator instantiation
        perform_experiments(model, scenarios, evaluator=evaluator)


def benchmark_model_evaluator_combinations(evaluator_classes, model_dict, scenarios_per_model, repeats=10):
    timing_results = {}
    print(f"Python version: {sys.version}")

    for model_name, model_function in model_dict.items():
        timing_results[model_name] = {}
        print(f"Running {model_name} with {scenarios_per_model[model_function]} scenarios")

        for evaluator_class in evaluator_classes:
            evaluator_name = evaluator_class.__name__
            print(f"Running {model_name} with {evaluator_name}")
            scenarios = scenarios_per_model[model_function]

            time_taken = timeit.repeat(
                lambda: run_model_with_evaluator(evaluator_class, model_function, scenarios),
                number=1,
                repeat=repeats)

            timing_results[model_name][evaluator_name] = time_taken

    return timing_results


if __name__ == '__main__':
    # Define evaluators and models to benchmark
    evaluators_to_test = [SequentialEvaluator, MultiprocessingEvaluator, MPIEvaluator]

    model_dict = {
        'python': get_python_model_instance,
        'lake': get_lake_model_instance,
        'flu': get_flu_model_instance
    }

    scenarios_per_model = {
        get_python_model_instance: 10000,
        get_lake_model_instance: 1000,
        get_flu_model_instance: 250
    }

    results = benchmark_model_evaluator_combinations(evaluators_to_test, model_dict, scenarios_per_model)

    # Save results
    with open("benchmark_results.pickle", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Print results
    for model, evaluators in results.items():
        for evaluator, time in evaluators.items():
            avg_time = sum(time) / len(time)
            print(f"{model} with {evaluator}: {avg_time:.2f} seconds")
