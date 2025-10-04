from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ie.types import FactType

from ..models import CandidateGroup, FactRecord, Partition, SimilarityPair


class UnionFind:
    def __init__(self, items: Iterable[int]) -> None:
        self._parent = {item: item for item in items}
        self._rank = {item: 0 for item in items}

    def find(self, item: int) -> int:
        parent = self._parent.get(item)
        if parent is None:
            self._parent[item] = item
            self._rank[item] = 0
            return item
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self._rank[root_a]
        rank_b = self._rank[root_b]
        if rank_a < rank_b:
            self._parent[root_a] = root_b
        elif rank_b < rank_a:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] += 1


class CandidateGrouper:
    def build_groups(
        self,
        partition: Partition,
        facts: Sequence[FactRecord],
        *,
        minhash_pairs: Sequence[SimilarityPair],
        embedding_pairs: Sequence[SimilarityPair],
    ) -> list[CandidateGroup]:
        fact_ids = {fact.id for fact in facts}
        if len(fact_ids) < 2:
            return []

        uf = UnionFind(fact_ids)
        for pair in minhash_pairs:
            uf.union(pair.source_id, pair.target_id)
        for pair in embedding_pairs:
            uf.union(pair.source_id, pair.target_id)

        components: dict[int, set[int]] = defaultdict(set)
        for fact_id in fact_ids:
            root = uf.find(fact_id)
            components[root].add(fact_id)

        groups: list[CandidateGroup] = []
        for members in components.values():
            if len(members) < 2:
                continue
            group = CandidateGroup(partition=partition, fact_ids=set(members))
            if minhash_pairs:
                group.add_similarity(
                    "minhash",
                    [pair for pair in minhash_pairs if pair.source_id in members and pair.target_id in members],
                )
            if embedding_pairs:
                group.add_similarity(
                    "embedding",
                    [pair for pair in embedding_pairs if pair.source_id in members and pair.target_id in members],
                )
            groups.append(group)

        return groups
