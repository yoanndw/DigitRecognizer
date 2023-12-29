from typing import List

from dataset import Dataset

def _compute_prior(dataset: Dataset, target: int) -> float: # P(Y)
    targets = dataset.target
    return len([t for t in targets if t == target]) / len(targets)


def _compute_likelihood_at_index(dataset: Dataset, i: int, expected_value: int, target: int) -> float:
    count_values_with_freeman_target = len([f for i_target, f in enumerate(dataset.freeman) if f[i] == expected_value and dataset.target[i_target] == target])
    count_values_with_target = len([t for t in dataset.target if t == target])

    return (count_values_with_freeman_target + 1) / (count_values_with_target + 10)


def _compute_likelihood(dataset: Dataset, expected_freeman: List[int], target: int) -> float:
    likelihood = 1
    for (i, f) in enumerate(expected_freeman):
        try:
            prob = _compute_likelihood_at_index(dataset, i, f, dataset.target[i])
        except IndexError:
            # Happens when a freeman code is longer than another one
            # In this case: do not impact likelihood
            prob = 1
        
        likelihood *= prob

    return likelihood


def _compute_freeman_prob_at_i(dataset: Dataset, i: int, expected_value: int) -> float:
    count_values_with_freeman = len([f for f in dataset.freeman if f[i] == expected_value])
    return count_values_with_freeman / len(dataset.freeman)


def _compute_freeman_prob(dataset: Dataset, expected_freeman: List[int]) -> float:
    product = 1
    for (i, f) in enumerate(expected_freeman):
        try:
            prob = _compute_freeman_prob_at_i(dataset, i, f)
        except IndexError:
            # Happens when a freeman code is longer than another one
            # In this case: do not impact the result
            prob = 1
        finally:
            product *= prob

    return product


def compute_posterior(dataset: Dataset, freeman: List[int], target: int) -> float:
    likelihood = _compute_likelihood(dataset, freeman, target)
    prior = _compute_prior(dataset, target)
    freeman_prob = _compute_freeman_prob(dataset, freeman)

    return likelihood * prior / freeman_prob
