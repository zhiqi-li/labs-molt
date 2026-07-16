"""Dependency-light protocol for an exhausted RL rollout producer.

The producer can discover that a stateful dataloader is exhausted only after
the ``StopIteration`` probe following an exact full final batch.  By then the
last optimizer batch is already queued.  This module defines the small queue
message that carries the *post-probe* client state to the training actor so it
can persist a terminal checkpoint at the already-completed global step.
"""

from copy import deepcopy
from typing import Any, Callable, Dict


TERMINAL_CLIENT_STATE = "terminal_client_state"


def _validated_terminal_client_states(client_states: Dict[str, Any]) -> Dict[str, Any]:
    """Return an isolated terminal state or reject a premature terminal signal."""
    if not isinstance(client_states, dict):
        raise TypeError("terminal client states must be a dict")
    loader_state = client_states.get("data_loader_state_dict")
    if not isinstance(loader_state, dict):
        raise RuntimeError("terminal client states require a dataloader state dict")
    if not bool(loader_state.get("_iterator_finished", False)):
        raise RuntimeError("terminal client states require an exhausted dataloader iterator")
    return deepcopy(client_states)


def make_terminal_payload(client_states: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Freeze exhausted client state in a queue-safe terminal payload."""
    return TERMINAL_CLIENT_STATE, _validated_terminal_client_states(client_states)


def terminal_client_states_from_payload(payload: Any) -> Dict[str, Any] | None:
    """Decode a terminal payload, returning ``None`` for other queue messages."""
    if not (isinstance(payload, tuple) and len(payload) == 2 and payload[0] == TERMINAL_CLIENT_STATE):
        return None
    return _validated_terminal_client_states(payload[1])


def _same_terminal_frontier(left: Any, right: Dict[str, Any]) -> bool:
    """Compare only scalar progress markers, never opaque loader RNG payloads."""
    if not isinstance(left, dict):
        return False
    left_loader = left.get("data_loader_state_dict")
    right_loader = right.get("data_loader_state_dict")
    if not isinstance(left_loader, dict) or not isinstance(right_loader, dict):
        return False
    return bool(left_loader.get("_iterator_finished", False)) and all(
        left_value == right_value
        for left_value, right_value in (
            (left.get("global_step"), right.get("global_step")),
            (left.get("episode"), right.get("episode")),
            (left.get("total_consumed_prompts"), right.get("total_consumed_prompts")),
            (left_loader.get("_sampler_iter_yielded"), right_loader.get("_sampler_iter_yielded")),
        )
    )


def handle_terminal_payload(
    payload: Any,
    *,
    global_step: int,
    best_eval_metric_key: str,
    best_eval_metric_value: float,
    last_checkpoint_client_states: Dict[str, Any] | None,
    save_terminal_checkpoint: Callable[[int, Dict[str, Any]], None],
) -> Dict[str, Any] | None:
    """Finalize and, when needed, save one producer terminal message.

    Returning ``None`` means the queue payload belongs to another protocol.
    Otherwise the returned state is the authoritative latest checkpoint state.
    """
    client_states = terminal_client_states_from_payload(payload)
    if client_states is None:
        return None
    client_states["global_step"] = int(global_step)
    client_states["best_eval_metric_key"] = best_eval_metric_key
    client_states["best_eval_metric_value"] = best_eval_metric_value
    if not _same_terminal_frontier(last_checkpoint_client_states, client_states):
        save_terminal_checkpoint(int(global_step), client_states)
    return client_states
