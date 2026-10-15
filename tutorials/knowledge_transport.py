#!/usr/bin/env python3
"""Check which option knowledge survives a change of coordinates or behavior.

Python 3.10+, standard library only; run from any working directory:
  python3 /path/to/knowledge_transport.py       # machine-readable JSON
  python3 /path/to/knowledge_transport.py demo  # the same exact finite example
  python3 /path/to/knowledge_transport.py test  # deterministic mathematical checks

This is an original finite diagnostic, not a training or performance experiment.
The physical world and external rewards are the six-state delivery task from
learned_subgoal_cycle.py. No other file is imported. A route option stops on
arrival at A or C. Its only varying policy parameter is P(upper | S). Following
the option, a GIVEN continuation exits A for reward 1 or probes C for reward 2.
An option stopping at A/C does not terminate the physical task.

We compare an invertible recoding, a policy-version change with coordinates
fixed, and a lossy merge of A/C. All probabilities are specified, not learned.
Path enumeration and a separate finite transition-matrix calculation must agree.
The optional r - rho*tau + P*h score freezes rho and h; it is one backup, not
an average-reward gain solution for this episodic world.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import json

GAMMA = F(9, 10)
RHO = F(1, 5)
WORLD = {
    'S': {'upper': ('A', F(0)), 'lower': ('B', F(0))},
    'A': {'exit': ('X', F(1)), 'probe': ('Y', F(0))},
    'B': {'forward': ('C', F(0))},
    'C': {'exit': ('X', F(1)), 'probe': ('Y', F(2))},
    'X': {}, 'Y': {},
}
# Only A/C are queried by the route model. Terminal continuation is zero.
READOUT = {'S': F(0), 'A': F(1), 'B': F(0), 'C': F(2), 'X': F(0), 'Y': F(0)}
IDENTITY = {s: s for s in WORLD}
PERMUTATION = {'S': 'B', 'A': 'C', 'B': 'S', 'C': 'A', 'X': 'Y', 'Y': 'X'}


@dataclass(frozen=True)
class Path:
    probability: F
    states: tuple[str, ...]
    rewards: tuple[F, ...]
    physical_terminal: bool


@dataclass(frozen=True)
class Model:
    start: str
    reward: F                 # E[sum gamma^k R_(k+1)]
    total_reward: F           # E[sum R_(k+1)]
    duration: F               # E[number of primitive transitions]
    endpoint: dict[str, F]    # P(S_tau=s), including physical terminals
    kernel: dict[str, F]      # E[gamma^tau 1{S_tau=s, not terminal}]


def policy(p: F) -> dict:
    if not 0 <= p <= 1:
        raise ValueError('route probability must lie in [0, 1]')
    return {'S': {'upper': p, 'lower': 1-p}, 'A': {'exit': F(1)},
            'B': {'forward': F(1)}, 'C': {'probe': F(1)}}


def enumerate_paths(p: F, complete: bool = False) -> list[Path]:
    """Walk actual finite paths. No option matrix or model supplies these rows."""
    pi, stops, paths = policy(p), set() if complete else {'A', 'C'}, []

    def visit(states, rewards, probability):
        source = states[-1]
        if rewards and (not WORLD[source] or source in stops):
            paths.append(Path(probability, states, rewards, not WORLD[source]))
            return
        for action, chance in pi[source].items():
            if chance:
                successor, reward = WORLD[source][action]
                visit(states + (successor,), rewards + (reward,), probability * chance)

    visit(('S',), (), F(1))
    return paths


def from_paths(paths: list[Path]) -> Model:
    endpoints, kernel = {}, {}
    reward = total = duration = F(0)
    for path in paths:
        prob, end, length = path.probability, path.states[-1], len(path.rewards)
        reward += prob * sum((GAMMA**k * r for k, r in enumerate(path.rewards)), F(0))
        total += prob * sum(path.rewards, F(0))
        duration += prob * length
        endpoints[end] = endpoints.get(end, F(0)) + prob
        if not path.physical_terminal:
            kernel[end] = kernel.get(end, F(0)) + prob * GAMMA**length
    return Model(paths[0].states[0], reward, total, duration, endpoints, kernel)


def matrix_model(world: dict, pi: dict, stops: set, start: str) -> Model:
    """Sum finite powers of the continuation matrix, independently of paths.

    T carries mass that keeps executing the option; E carries stopping mass.
    Rewards on the stopping transition are retained. Physical terminal states
    remain in E but are removed from the discounted continuation kernel.
    """
    states = tuple(world)
    n, index = len(states), {s: i for i, s in enumerate(states)}
    transition = [[F(0) for _ in states] for _ in states]
    ending = [[F(0) for _ in states] for _ in states]
    immediate = [F(0) for _ in states]
    for source, actions in pi.items():
        i = index[source]
        if source in stops:
            continue
        if sum(actions.values(), F(0)) != 1:
            raise ValueError('policy row must sum to one')
        for action, chance in actions.items():
            successor, reward = world[source][action]
            j = index[successor]
            immediate[i] += chance * reward
            target = ending if successor in stops or not world[successor] else transition
            target[i][j] += chance

    mass = [F(int(s == start)) for s in states]
    endpoint, kernel = [F(0) for _ in states], [F(0) for _ in states]
    reward = total = duration = F(0)
    # The given directed acyclic world has at most three primitive transitions.
    for t in range(n):
        if not any(mass):
            break
        one_reward = sum((mass[i] * immediate[i] for i in range(n)), F(0))
        reward += GAMMA**t * one_reward
        total += one_reward
        duration += sum(mass, F(0))
        for j, successor in enumerate(states):
            stopping = sum((mass[i] * ending[i][j] for i in range(n)), F(0))
            endpoint[j] += stopping
            if world[successor]:
                kernel[j] += GAMMA**(t+1) * stopping
        mass = [sum((mass[i] * transition[i][j] for i in range(n)), F(0)) for j in range(n)]
    if any(mass):
        raise ValueError('this finite diagnostic requires an acyclic terminating option')
    return Model(start, reward, total, duration,
                 {s: endpoint[i] for i, s in enumerate(states) if endpoint[i]},
                 {s: kernel[i] for i, s in enumerate(states) if kernel[i]})


def discounted_backup(model: Model, values: dict) -> F:
    # Gamma^tau is already in the kernel: do not multiply by gamma again.
    return model.reward + sum((weight * values[s] for s, weight in model.kernel.items()), F(0))


def fixed_rate_backup(model: Model, h: dict, rho: F = RHO) -> F:
    """Evaluate a given rate/bias pair, without solving an average-reward task.

    This diagnostic's episodic boundary uses h(X)=h(Y)=0. The function does not
    impose terminal zeros: its caller supplies them in h, including after state
    recoding. Endpoint P includes terminal mass. A continuing problem needs its
    own state, boundary and gain assumptions; this formula alone supplies none.
    """
    # These are the UNDISCOUNTED reward and endpoint probability, not the kernel.
    return model.total_reward - rho * model.duration + sum(
        (prob * h[s] for s, prob in model.endpoint.items()), F(0))


def transport(model: Model, mapping: dict) -> Model:
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('exact coordinate transport requires a bijection')
    return Model(mapping[model.start], model.reward, model.total_reward, model.duration,
                 {mapping[s]: p for s, p in model.endpoint.items()},
                 {mapping[s]: p for s, p in model.kernel.items()})


def pushforward(values: dict, mapping: dict) -> dict:
    result = {}
    for state, weight in values.items():
        encoded = mapping[state]
        result[encoded] = result.get(encoded, F(0)) + weight
    return result


def view(model: Model, readout: dict) -> dict:
    q, rate_q = discounted_backup(model, readout), fixed_rate_backup(model, readout)
    return dict(start=model.start, external_discounted_reward=model.reward,
                external_total_reward=model.total_reward, expected_duration=model.duration,
                endpoint_probability=model.endpoint, discounted_endpoint_kernel=model.kernel,
                continuation_readout=readout, discounted_backup=q, fixed_rate_backup=rate_q,
                exact=dict(discounted_backup=str(q), fixed_rate_backup=str(rate_q),
                           expected_duration=str(model.duration)))


def calculate() -> dict:
    old = from_paths(enumerate_paths(F(3, 4)))
    changed = from_paths(enumerate_paths(F(1, 4)))
    mild = from_paths(enumerate_paths(F(2, 3)))
    reference = from_paths(enumerate_paths(F(1, 2), complete=True))
    recoded = transport(old, PERMUTATION)
    recoded_readout = {PERMUTATION[s]: v for s, v in READOUT.items()}
    merge = {s: ('Z' if s in {'A', 'C'} else s) for s in WORLD}
    projected = Model(old.start, old.reward, old.total_reward, old.duration,
                      pushforward(old.endpoint, merge), pushforward(old.kernel, merge))
    # 3/2 minimizes max(|c-V(A)|, |c-V(C)|), not error at one chosen S backup.
    projected_readout = {merge[s]: v for s, v in READOUT.items() if s not in {'A', 'C'}}
    projected_readout['Z'] = F(3, 2)
    fitted_single_query = discounted_backup(old, READOUT) / sum(old.kernel.values(), F(0))
    reference_q = discounted_backup(reference, READOUT)

    def decision(model):
        q = discounted_backup(model, READOUT)
        return dict(route=q, reference=reference_q,
                    greedy='route' if q > reference_q else 'reference',
                    gap=abs(q-reference_q))

    scenarios = {
        'bijective_recoding': dict(
            changed_object='coordinate labels only; physical policy unchanged',
            mapping=PERMUTATION, physical_true=view(old, READOUT), old=view(old, READOUT),
            transported=view(recoded, recoded_readout),
            stale_readout_backup=discounted_backup(recoded, READOUT)),
        'policy_change': dict(
            changed_object='P(upper|S): 3/4 to 1/4; coordinates unchanged',
            mapping=IDENTITY, physical_true=view(changed, READOUT), old=view(old, READOUT),
            transported=view(transport(old, IDENTITY), READOUT),
            endpoint_total_variation=F(1, 2),
            backup_error=discounted_backup(changed, READOUT)-discounted_backup(old, READOUT),
            old_decision=decision(old), current_decision=decision(changed)),
        'lossy_merge': dict(
            changed_object='A and C have the same coordinate; physical policy unchanged',
            mapping=merge, physical_true=view(old, READOUT), old=view(old, READOUT),
            transported=dict(status='lossy projection; no exact inverse transport',
                             **view(projected, projected_readout)),
            merged_states=['A', 'C'], physical_readouts={'A': F(1), 'C': F(2)},
            minimax_readout=F(3, 2), unavoidable_max_readout_error=F(1, 2),
            probe_external_rewards={'A': F(0), 'C': F(2)},
            unavoidable_max_probe_reward_error=F(1),
            one_query_fitted_readout=fitted_single_query,
            one_query_fitted_backup=fitted_single_query * sum(old.kernel.values(), F(0)),
            note='Matching one S backup does not reconstruct the two state-conditional answers.'),
        'error_without_decision_change': dict(
            changed_object='P(upper|S): 3/4 to 2/3; coordinates unchanged',
            physical_true=view(mild, READOUT), old=view(old, READOUT),
            transported=view(transport(old, IDENTITY), READOUT),
            endpoint_total_variation=F(1, 12),
            backup_error=discounted_backup(mild, READOUT)-discounted_backup(old, READOUT),
            old_decision=decision(old), current_decision=decision(mild)),
    }
    return dict(
        schema='crl-knowledge-transport-v1', evidence_kind='exact_finite_diagnostic',
        gamma=GAMMA, rho=RHO, states=list(WORLD),
        units=dict(reward='external reward units', discounted_backup='external reward units',
                   continuation_readout='external reward units',
                   duration='primitive transitions per option call',
                   endpoint_probability='probability per option call',
                   discounted_endpoint_kernel='dimensionless discounted probability',
                   rho='external reward units per primitive transition',
                   fixed_rate_backup='external reward units'),
        fixed_rate_scope='One r-rho*tau+P*h backup with given rho=1/5 and h=READOUT, '
                         'including h(X)=h(Y)=0 at this episodic boundary; no gain is solved.',
        assumptions=['Known finite physical dynamics; specified option policies, no learning.',
                     'Same physical world and external reward in every case.',
                     'Option stops at A/C; physical episodes stop only at X/Y.',
                     'Choice is compared only between two given options with fixed continuation.',
                     'A lossy projected backup is a prediction, not a new executed policy return.',
                     'The reference option completes the fixed continuation with P(upper|S)=1/2.'],
        reference=view(reference, READOUT), scenarios=scenarios,
        paths={name: [dict(probability=p.probability, states=p.states, rewards=p.rewards,
                          duration=len(p.rewards), physical_terminal=p.physical_terminal)
                      for p in enumerate_paths(prob, complete)]
               for name, prob, complete in [('old', F(3, 4), False),
                   ('current', F(1, 4), False), ('mild', F(2, 3), False),
                   ('reference', F(1, 2), True)]})


def check() -> dict:
    checks = []
    for p in (F(0), F(1, 4), F(1, 2), F(2, 3), F(3, 4), F(1)):
        for complete in (False, True):
            enumerated = from_paths(enumerate_paths(p, complete))
            matrix = matrix_model(WORLD, policy(p), set() if complete else {'A', 'C'}, 'S')
            assert enumerated == matrix, (p, complete, enumerated, matrix)
            assert sum(matrix.endpoint.values(), F(0)) == 1
    checks.append('12 independent path enumeration / matrix expectation comparisons')
    old = from_paths(enumerate_paths(F(3, 4)))
    assert transport(old, IDENTITY) == old
    transformed = transport(old, PERMUTATION)
    inverse = {v: k for k, v in PERMUTATION.items()}
    assert transport(transformed, inverse) == old
    readout = {PERMUTATION[s]: v for s, v in READOUT.items()}
    transformed_world = {PERMUTATION[s]: {a: (PERMUTATION[n], r) for a, (n, r) in row.items()}
                         for s, row in WORLD.items()}
    transformed_policy = {PERMUTATION[s]: row for s, row in policy(F(3, 4)).items()}
    assert matrix_model(transformed_world, transformed_policy,
                        {PERMUTATION[s] for s in {'A', 'C'}}, PERMUTATION['S']) == transformed
    assert discounted_backup(transformed, readout) == discounted_backup(old, READOUT)
    assert fixed_rate_backup(transformed, readout) == fixed_rate_backup(old, READOUT)
    checks.append('identity, inverse, whole-world permutation, and both backup invariances')
    data = calculate()
    cases = data['scenarios']
    assert cases['bijective_recoding']['stale_readout_backup'] == F(621, 400)
    checks.append('a transported model with an untransported readout gives the wrong answer')
    for key, truth in [('policy_change', F(36, 25)), ('error_without_decision_change', F(57, 50))]:
        assert cases[key]['physical_true']['discounted_backup'] == truth
        assert cases[key]['transported']['discounted_backup'] == F(27, 25)
    assert cases['policy_change']['old_decision']['greedy'] == 'reference'
    assert cases['policy_change']['current_decision']['greedy'] == 'route'
    mild = cases['error_without_decision_change']
    assert mild['old_decision']['greedy'] == mild['current_decision']['greedy'] == 'reference'
    assert F(0) < mild['backup_error'] < mild['old_decision']['gap']
    checks.append('behavioral drift survives coordinate transport; a small error preserves the action margin')
    merged = cases['lossy_merge']
    try:
        transport(old, merged['mapping'])
    except ValueError:
        pass
    else:
        raise AssertionError('a many-to-one map cannot have inverse coordinate transport')
    assert merged['physical_readouts']['A'] != merged['physical_readouts']['C']
    assert merged['probe_external_rewards']['A'] != merged['probe_external_rewards']['C']
    assert merged['one_query_fitted_backup'] == merged['physical_true']['discounted_backup']
    assert merged['transported']['discounted_backup'] == F(1053, 800)
    checks.append('irreversible merge loses conditional answers even when one aggregate backup can be fitted')
    assert cases['policy_change']['physical_true']['fixed_rate_backup'] == F(7, 5)
    assert cases['policy_change']['old']['fixed_rate_backup'] == F(1)
    assert data['reference']['fixed_rate_backup'] == F(1)
    checks.append('fixed-rho one-backup units and terminal zero continuation')
    return dict(status='pass', evidence_kind='exact_finite_diagnostic', checks=checks,
                limitation='These checks do not train a joint learner or measure a method advantage.')


def json_value(value):
    if isinstance(value, F):
        return float(value)
    raise TypeError(type(value).__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('demo', 'test'), nargs='?', default='demo')
    args = parser.parse_args()
    print(json.dumps(check() if args.command == 'test' else calculate(),
                     default=json_value, ensure_ascii=False, indent=2))
