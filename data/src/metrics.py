"""Ranking metrics with the assignment's min(k, relevant-item-count) denominator."""


def average_precision_at_k(relevant, predicted, k=5):
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    seen = set()
    for rank, item in enumerate(predicted[:k], 1):
        if item in relevant and item not in seen:
            hits += 1
            total += hits / rank
        seen.add(item)
    return total / min(k, len(relevant))


def mean_average_precision_at_k(relevant_by_user, predicted_by_user, users, k=5):
    users = list(users)
    if not users:
        raise ValueError("The evaluation cohort must not be empty")
    # Deliberately require a prediction for every evaluated user.
    return sum(average_precision_at_k(relevant_by_user.get(user, set()),
                                    predicted_by_user[user], k) for user in users) / len(users)
