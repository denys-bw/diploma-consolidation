from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from consolidator.linkage.decision import DecisionRule, MatchStatus

IDENTIFIABLE_ATTRIBUTES = 3
EVENTS_PER_PARAMETER = 10
FEW_ATTRIBUTES = "few_attributes"
FALSE_MATCHES = "false_matches"
MISSED_MATCHES = "missed_matches"


@dataclass(frozen=True)
class Reliability:
    accepted: int
    possible: int
    contested: int
    expected_false: float
    expected_missed: float
    expected_matches: float
    false_match_rate: float
    missed_match_rate: float
    training_rounds: int
    one_to_one: bool
    attributes: int
    corrected: bool = False

    @property
    def false_share(self) -> float:
        return self.expected_false / self.accepted if self.accepted else 0.0

    @property
    def missed_share(self) -> float:
        return self.expected_missed / self.expected_matches if self.expected_matches > 0 else 0.0

    @property
    def identifiable(self) -> bool:
        return self.attributes >= IDENTIFIABLE_ATTRIBUTES

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.identifiable and not self.corrected:
            reasons.append(FEW_ATTRIBUTES)
        if self.expected_false > max(self.false_match_rate * self.accepted, 1.0):
            reasons.append(FALSE_MATCHES)
        if self.expected_missed > max(self.missed_match_rate * self.expected_matches, 1.0):
            reasons.append(MISSED_MATCHES)
        return tuple(reasons)

    @property
    def reliable(self) -> bool:
        return not self.reasons

    @property
    def suggested_reviews(self) -> int:
        if FEW_ATTRIBUTES in self.reasons:
            return EVENTS_PER_PARAMETER * (self.attributes + 1)
        return self.possible if self.reasons else 0


def assess(
    probabilities: np.ndarray,
    independent: np.ndarray,
    statuses: np.ndarray,
    decision: DecisionRule,
    training_rounds: int,
    one_to_one: bool,
    attributes: int,
    corrected: bool = False,
) -> Reliability:
    accepted = decision.accepted(probabilities)
    contested = accepted != decision.accepted(independent)
    errors = decision.errors(probabilities, contested)
    return Reliability(
        accepted=int(accepted.sum()),
        possible=int((statuses == MatchStatus.POSSIBLE).sum()),
        contested=int(contested.sum()),
        expected_false=float(errors[accepted].sum()),
        expected_missed=float(errors[~accepted].sum()),
        expected_matches=float(probabilities.sum()),
        false_match_rate=decision.false_match_rate,
        missed_match_rate=decision.missed_match_rate,
        training_rounds=training_rounds,
        one_to_one=one_to_one,
        attributes=attributes,
        corrected=corrected,
    )
