"""Experiment defaults and the test-time budgets every condition is scored under."""

# test-time budgets: (max retrieval hops, memories read per hop)
TEST_CONFIGS = [(1, 2), (1, 3), (1, 5), (2, 3), (2, 5)]
HEADLINE_STRICT = (1, 3)  # one retrieval round
HEADLINE_GENEROUS = (2, 3)  # two retrieval rounds

DEFAULTS = dict(
    n_chains=24,
    n_sessions=5,
    idle_tokens_per_session=2400,
    amplitude=0.5,
    hallucination_rate=0.3,
    verifier_miss=0.15,
    retrieval_noise=0.04,
)

# Memory and background-thinking upgrades. All off by default, so results stay comparable.
UPGRADES_OFF = dict(trial_links=False, verify_on_use=False, trial_expiry=0, skip_unlinked=False, adaptive_noise=False)
ALL_UPGRADES = dict(trial_links=True, verify_on_use=True, trial_expiry=60, skip_unlinked=True, adaptive_noise=True)

UPGRADE_ROWS = [
    ("C as reported", {}),
    ("+ trial period for new links", dict(trial_links=True)),
    ("+ check link on use", dict(verify_on_use=True)),
    ("+ expire unconfirmed links", dict(trial_expiry=60)),
    ("+ skip unlinked seeds", dict(skip_unlinked=True)),
    ("+ adaptive noise", dict(adaptive_noise=True)),
    ("all upgrades", ALL_UPGRADES),
]
