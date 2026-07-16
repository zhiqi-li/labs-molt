import sys
import types
from types import SimpleNamespace

import pytest

if "vllm" not in sys.modules:
    fake_vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_vllm.SamplingParams = SamplingParams
    sys.modules["vllm"] = fake_vllm

from molt.trainer.terminal_checkpoint import (
    handle_terminal_payload,
    make_terminal_payload,
    terminal_client_states_from_payload,
)


class ExactFullBatchIterator:
    """Mimic StatefulDataLoader's exact-boundary exhaustion bookkeeping."""

    def __init__(self, size):
        self.size = size
        self.yielded = 0
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.yielded == self.size:
            self.finished = True
            raise StopIteration
        self.yielded += 1
        return self.yielded

    def state_dict(self):
        return {
            "_sampler_iter_yielded": self.yielded,
            "_iterator_finished": self.finished,
        }


def test_exact_full_batch_terminal_payload_uses_post_stop_iteration_state():
    loader = ExactFullBatchIterator(776)
    for _ in range(776):
        next(loader)

    # Consuming exactly the final item is not enough for a stateful iterator to
    # know that it is exhausted. This is the stale state old rolling checkpoints
    # captured after the final optimizer batch.
    assert loader.state_dict() == {
        "_sampler_iter_yielded": 776,
        "_iterator_finished": False,
    }
    with pytest.raises(StopIteration):
        next(loader)

    payload = make_terminal_payload(
        {
            "episode": 2,
            "total_consumed_prompts": 2328,
            "data_loader_state_dict": loader.state_dict(),
            "rollout_generator_state_dict": {},
        }
    )
    state = terminal_client_states_from_payload(payload)

    assert state == {
        "episode": 2,
        "total_consumed_prompts": 2328,
        "data_loader_state_dict": {
            "_sampler_iter_yielded": 776,
            "_iterator_finished": True,
        },
        "rollout_generator_state_dict": {},
    }


def test_terminal_payload_is_isolated_and_rejects_unfinished_iterator():
    state = {
        "episode": 0,
        "data_loader_state_dict": {"_iterator_finished": True},
    }
    payload = make_terminal_payload(state)
    state["data_loader_state_dict"]["_iterator_finished"] = False
    assert terminal_client_states_from_payload(payload)["data_loader_state_dict"]["_iterator_finished"] is True

    with pytest.raises(RuntimeError, match="exhausted"):
        make_terminal_payload(state)
    assert terminal_client_states_from_payload("done") is None


def test_terminal_handler_saves_completed_step_once_with_exact_client_state():
    payload = make_terminal_payload(
        {
            "episode": 2,
            "total_consumed_prompts": 2328,
            "data_loader_state_dict": {
                "_sampler_iter_yielded": 776,
                "_iterator_finished": True,
            },
            "rollout_generator_state_dict": {},
        }
    )
    saves = []

    state = handle_terminal_payload(
        payload,
        global_step=291,
        best_eval_metric_key="eval_reward",
        best_eval_metric_value=0.25,
        last_checkpoint_client_states=None,
        save_terminal_checkpoint=lambda step, client_state: saves.append((step, client_state.copy())),
    )

    assert saves == [(291, state)]
    assert state["global_step"] == 291
    assert state["episode"] == 2
    assert state["total_consumed_prompts"] == 2328
    assert state["data_loader_state_dict"] == {
        "_sampler_iter_yielded": 776,
        "_iterator_finished": True,
    }

    # A short final batch can already have persisted the same terminal state on
    # its normal cadence; the terminal message must not rewrite it needlessly or
    # compare opaque loader/generator payloads (which may contain tensors).
    class OpaqueState:
        def __eq__(self, _other):
            raise AssertionError("opaque state must not participate in frontier comparison")

    last_checkpoint = dict(state)
    last_checkpoint["rollout_generator_state_dict"] = {"opaque": OpaqueState()}
    handle_terminal_payload(
        payload,
        global_step=291,
        best_eval_metric_key="eval_reward",
        best_eval_metric_value=0.25,
        last_checkpoint_client_states=last_checkpoint,
        save_terminal_checkpoint=lambda step, client_state: saves.append((step, client_state.copy())),
    )
    assert saves == [(291, state)]


def _rl_trainer_or_skip():
    try:
        import molt.trainer.rl_trainer as rl_trainer
    except ModuleNotFoundError:
        pytest.skip("full Ray/torchdata runtime is not installed in the unit-test environment")
    return rl_trainer


def test_generate_actor_emits_exact_full_batch_post_probe_terminal_state():
    rl_trainer = _rl_trainer_or_skip()
    loader = ExactFullBatchIterator(776)
    loader.yielded = 776

    class PromptLoader:
        sampler = object()

        def __len__(self):
            return 776

        def state_dict(self):
            return loader.state_dict()

    class Generator:
        calls = 0

        def generate_samples(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return [object()], {}, 8, False
            with pytest.raises(StopIteration):
                next(loader)
            return [], {}, 0, True

        def state_dict(self):
            return {}

    class Queue:
        def __init__(self, values=None):
            self.values = list(values or [])

        def get(self, block=True):
            assert block
            return self.values.pop(0)

        def put(self, value, block=True):
            assert block
            self.values.append(value)

    rollout_queue = Queue()
    actor = SimpleNamespace(
        args=SimpleNamespace(
            eval=SimpleNamespace(steps=float("inf")),
            train=SimpleNamespace(
                num_episodes=3,
                force_on_policy=True,
                rollout_replay_dir=None,
                rollout_dump_dir=None,
            ),
        ),
        _eval_only=False,
        _eval_at_start=False,
        _partial_rollout=True,
        _last_eval_step=-1,
        _next_eval_step=float("inf"),
        eval_dataloader=None,
        prompts_dataloader=PromptLoader(),
        samples_generator=Generator(),
        rollout_slots=Queue([290, 291]),
        rollout_queue=rollout_queue,
        generate_kwargs={},
    )
    metadata = getattr(rl_trainer.GenerateSamplesActor, "__ray_metadata__", None)
    actor_class = getattr(metadata, "modified_class", rl_trainer.GenerateSamplesActor)

    actor_class.fit(actor, episode=2, total_consumed_prompts=2320)

    assert len(rollout_queue.values) == 3
    batch, terminal, done = rollout_queue.values
    assert len(batch[0]) == 1
    assert batch[1]["data_loader_state_dict"]["_iterator_finished"] is False
    terminal_state = terminal_client_states_from_payload(terminal)
    assert terminal_state["episode"] == 2
    assert terminal_state["total_consumed_prompts"] == 2328
    assert terminal_state["data_loader_state_dict"] == {
        "_sampler_iter_yielded": 776,
        "_iterator_finished": True,
    }
    assert done == "done"


def test_training_actor_saves_terminal_state_at_completed_global_step():
    rl_trainer = _rl_trainer_or_skip()
    terminal = make_terminal_payload(
        {
            "episode": 2,
            "total_consumed_prompts": 2328,
            "data_loader_state_dict": {
                "_sampler_iter_yielded": 776,
                "_iterator_finished": True,
            },
            "rollout_generator_state_dict": {},
        }
    )
    payloads = [([object()], {}, {}, 0.0, 0.0), terminal, "done"]
    saved = []

    class PayloadQueue:
        def get(self, block=True):
            assert block
            return payloads.pop(0)

    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(force_on_policy=False)),
        rollout_queue=PayloadQueue(),
        rollout_slots=SimpleNamespace(put=lambda *_args, **_kwargs: None),
        vllm_engines=[],
        best_eval_metric_key="eval_reward",
        best_eval_metric_value=0.25,
        wandb_logger=None,
        tensorboard_logger=None,
        train_step=lambda _samples, step: ({}, step + 1),
        save_logs_and_checkpoints=lambda *_args, **_kwargs: False,
        save_terminal_checkpoint=lambda step, state: saved.append((step, state.copy())),
    )
    metadata = getattr(rl_trainer.TrainingActor, "__ray_metadata__", None)
    actor_class = getattr(metadata, "modified_class", rl_trainer.TrainingActor)

    actor_class.fit(trainer, global_step=290)

    assert len(saved) == 1
    step, state = saved[0]
    assert step == 291
    assert state["global_step"] == 291
    assert state["episode"] == 2
    assert state["total_consumed_prompts"] == 2328
    assert state["data_loader_state_dict"]["_sampler_iter_yielded"] == 776
    assert state["data_loader_state_dict"]["_iterator_finished"] is True
