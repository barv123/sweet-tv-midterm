"""Independent content-cosine and directional association-rule methods.

These functions receive training matrices and exogenous item metadata only.
Validation relevance never enters either fitting or scoring interface.
"""

from collections import Counter
import math
import re

import numpy as np
from scipy import sparse


def l2_rows(matrix):
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms, dtype=float), where=norms > 0)
    return sparse.diags(inverse) @ matrix


def ranked_indices(scores, candidate_ids, popularity, limit=100):
    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Nonfinite recommendation score")
    rounded = np.round(scores, 12)
    # Zero-evidence rows reduce exactly to the established popularity fallback.
    return np.lexsort((candidate_ids, -popularity, -rounded))[:limit]


def metadata_tokens(record, include_title):
    tokens = {}
    for field, weight in (("category", 1.0), ("director", 0.5)):
        value = record.get(field)
        tokens[f"{field}={value.casefold().strip()}" if value else f"missing_{field}"] = weight if value else 0.1
    genres = {record.get(field) for field in ("genre1", "genre2", "genre3")} - {None, ""}
    if genres:
        tokens.update({"genre=" + genre.casefold().strip(): 1.0 for genre in genres})
    else:
        tokens["missing_genre"] = 0.1
    year = re.match(r"^(19\d\d|20[01]\d|2020)", record.get("production_year") or "")
    tokens["decade=" + str(int(year[0]) // 10 * 10) if year else "missing_year"] = 0.3 if year else 0.1
    if include_title:
        title = record.get("title") or ""
        quoted = re.search(r"'([^']+)'", title)
        text = quoted[1] if quoted else title
        words = {word for word in re.findall(r"[^\W\d_]{2,}", text.casefold()) if len(word) >= 3}
        tokens.update({"title=" + word: 1.0 for word in words})
        if not words:
            tokens["missing_title"] = 0.1
    return tokens


def content_features(metadata, item_ids, include_title):
    """Fit vocabulary/IDF on pre-cutoff metadata; transform target-only metadata."""
    documents = {item: metadata_tokens(record, include_title) for item, record in metadata.items()}
    fitting = [documents[item] for item, record in metadata.items() if record["metadata_period"] == 0]
    frequencies = Counter(token for document in fitting for token in document)
    vocabulary = {token: index for index, token in enumerate(sorted(frequencies))}
    idf = {token: 1.0 + math.log((1 + len(fitting)) / (1 + count)) for token, count in frequencies.items()}
    rr, cc, vv = [], [], []
    for index, item in enumerate(item_ids):
        for token, weight in documents.get(int(item), {}).items():
            if token in vocabulary:
                rr.append(index)
                cc.append(vocabulary[token])
                vv.append(weight * idf[token])
    matrix = sparse.csr_matrix((vv, (rr, cc)), shape=(len(item_ids), len(vocabulary)), dtype=np.float64)
    return l2_rows(matrix).tocsr(), {"feature_count": len(vocabulary), "idf_fit_documents": len(fitting),
        "include_title_tokens": include_title, "nonzero_item_vectors": int(np.sum(matrix.getnnz(axis=1) > 0)),
        "idf_uses_target_only_metadata": False}


def content_score_blocks(user_history, item_features, candidate_columns, batch_size=32):
    target = item_features[candidate_columns]
    for first in range(0, user_history.shape[0], batch_size):
        profiles = l2_rows(user_history[first:first + batch_size] @ item_features)
        yield first, (profiles @ target.T).toarray()


def mine_association_rules(binary_history, item_ids, candidate_columns, min_pair_count=25,
                           min_lift=1.0, min_confidence=0.1, min_item_viewers=20, max_rules=100):
    """One transaction per training user; bounded sparse co-occurrence blocks."""
    history = binary_history.astype(np.int32).tocsc()
    n_users = history.shape[0]
    counts = np.asarray(history.sum(axis=0)).ravel().astype(np.int64)
    eligible = np.zeros(len(item_ids), dtype=bool)
    eligible[candidate_columns] = True
    antecedents = np.flatnonzero(counts >= min_item_viewers)
    rr, cc, vv = [], [], []
    rule_examples = []
    accepted_before_cap = 0
    for first in range(0, len(antecedents), 128):
        subset = antecedents[first:first + 128]
        co_view = (history[:, subset].T @ history).tocsr()
        for row_index, antecedent in enumerate(subset):
            start, end = co_view.indptr[row_index:row_index + 2]
            consequents, together = co_view.indices[start:end], co_view.data[start:end]
            keep = ((consequents != antecedent) & eligible[consequents] & (counts[consequents] >= min_item_viewers)
                    & (together >= min_pair_count))
            consequents, together = consequents[keep], together[keep]
            confidence = together / counts[antecedent]
            lift = confidence / (counts[consequents] / n_users)
            keep = (confidence >= min_confidence) & (lift >= min_lift)
            consequents, together, confidence, lift = (values[keep] for values in (consequents, together, confidence, lift))
            weights = confidence * np.log1p(lift)
            order = np.lexsort((item_ids[consequents], -counts[consequents], -np.round(weights, 12)))
            accepted_before_cap += len(order)
            order = order[:max_rules]
            for position in order:
                consequent = int(consequents[position])
                rr.append(int(antecedent))
                cc.append(consequent)
                vv.append(float(weights[position]))
                # A deterministic, inspectable sample; complete rule weights stay in the private cache.
                if len(rule_examples) < 30 or (item_ids[antecedent] == 2400480 and len(rule_examples) < 130):
                    rule_examples.append({"antecedent": int(item_ids[antecedent]), "consequent": int(item_ids[consequent]),
                        "users_both": int(together[position]), "antecedent_viewers": int(counts[antecedent]),
                        "consequent_viewers": int(counts[consequent]), "transactions": n_users,
                        "support": float(together[position] / n_users), "confidence": float(confidence[position]),
                        "lift": float(lift[position]), "weight": float(weights[position])})
    rules = sparse.csr_matrix((vv, (rr, cc)), shape=(len(item_ids), len(item_ids)), dtype=np.float64)
    return rules, {"transactions": n_users, "frequent_antecedents": len(antecedents),
        "min_pair_count": min_pair_count, "min_lift": min_lift, "min_confidence": min_confidence,
        "min_item_viewers": min_item_viewers, "max_rules_per_antecedent": max_rules,
        "rules_before_cap": accepted_before_cap, "retained_rules": rules.nnz,
        "consequents_with_rules": len(set(cc)), "sample_rules": rule_examples}


def rule_score_blocks(user_history, rules, candidate_columns, batch_size=64):
    # Normalization does not change a user's ranking, but makes score scale explicit.
    sums = np.asarray(user_history.sum(axis=1)).ravel()
    inverse = np.divide(1.0, sums, out=np.zeros_like(sums), where=sums > 0)
    weighted = sparse.diags(inverse) @ user_history
    for first in range(0, weighted.shape[0], batch_size):
        yield first, (weighted[first:first + batch_size] @ rules[:, candidate_columns]).toarray()


def reciprocal_rank_fusion(rankings, candidate_ids, popularity, constant=60, limit=100):
    locations = {int(item): index for index, item in enumerate(candidate_ids)}
    score = np.zeros(len(candidate_ids))
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            score[locations[int(item)]] += 1.0 / (constant + rank)
    return candidate_ids[ranked_indices(score, candidate_ids, popularity, limit)].tolist()
