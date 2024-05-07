import numpy as np

# Modified https://github.com/google/rax/blob/1638551bf598de7e3d54a79ad56a3bddd8dc4546/rax/_src/metrics.py#L557

def opa_metric(
    scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
  r"""Ordered Pair Accuracy (OPA).

  Definition:

  .. math::
      \op{opa}(s, y) =
      \frac{1}{\sum_i \sum_j \II{y_i > y_j}}
      \sum_i \sum_j \II{s_i > s_j} \II{y_i > y_j}

  .. note::

    Pairs with equal labels (:math:`y_i = y_j`) are always ignored. Pairs with
    equal scores (:math:`s_i = s_j`) are considered incorrectly ordered.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.Array`, indicating the score of
      each item. Items for which the score is :math:`-\inf` are treated as
      unranked items.
    labels: A ``[..., list_size]``-:class:`~jax.Array`, indicating the relevance
      label for each item.

  Returns:
    The Ordered Pair Accuracy (OPA).
  """

  pair_label_diff = np.expand_dims(labels, -1) - np.expand_dims(labels, -2)
  pair_score_diff = np.expand_dims(scores, -1) - np.expand_dims(scores, -2)
  # Infer location of valid pairs through where
  correct_pairs = (pair_label_diff > 0) * (pair_score_diff > 0)
  # Calculate per list pairs.
  per_list_pairs = np.sum(
      pair_label_diff > 0, axis=(-2, -1)
  )
  # A workaround to bypass divide by zero.
  per_list_pairs = np.where(per_list_pairs == 0, 1, per_list_pairs)
  per_list_opa = np.divide(
      np.sum(
          correct_pairs,
          axis=(-2, -1),
      ),
      per_list_pairs,
  )
  return np.mean(per_list_opa)


if __name__ == "__main__":
  sz = 10
  scores = np.random.rand(sz)
  labels = np.random.randint(0, 10, size=(sz,))
  opa = opa_metric(scores, labels)
  print(opa)
  print(opa.shape)
  print(opa.item())
