import pickle
import sys

from ema_workbench import (MPIEvaluator, perform_experiments)
from models_to_benchmark import get_lake_model_instance
import timeit


def run_model_with_evaluator(evaluator_class, model_function, scenarios):
    model = model_function()
    with evaluator_class(model) as evaluator:  # Pass the **kwargs to the evaluator instantiation
        perform_experiments(model, scenarios, evaluator=evaluator)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        node_multiplier = float(sys.argv[1])
    else:
        node_multiplier = 1  # default value if not provided
        print("Warning: No node multiplier provided, using default value of 1")
    cpu_multiplier = 48 * node_multiplier

    n_scenarios = 100000
    print(f"Benchmarking {n_scenarios} Lake model iterations on DelftBlue with {node_multiplier} node(s) ({cpu_multiplier} CPU cores)")
    time_taken = timeit.repeat(
                lambda: run_model_with_evaluator(evaluator_class=MPIEvaluator, model_function=get_lake_model_instance, scenarios=n_scenarios),
                number=1,
                repeat=10)
    print(f"Time taken: {time_taken}")

    # Save results
    with open(f"benchmark_results_DelftBlue_scaling2_{node_multiplier}nodes.pickle", "wb") as handle:
        pickle.dump(time_taken, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Print results
    avg_time = sum(time_taken) / len(time_taken)
    print(f"{avg_time:.3f} seconds, {n_scenarios / avg_time:.2f} scenarios per second")
