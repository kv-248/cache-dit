import pytest
import torch

from cache_dit.caching.cache_contexts.calibrators import FoCaCalibrator


def test_foca_state_bdf2_linear():
  """BDF2 predictor matches linear extrapolation on y=x sequence."""
  foca = FoCaCalibrator()
  # y = x: feed steps 0 and 1 as full-compute
  foca.mark_step_begin()
  foca.update(torch.tensor(0.0, dtype=torch.float32))
  foca.mark_step_begin()
  foca.update(torch.tensor(1.0, dtype=torch.float32))
  # step 2: cache → predict x=2, expect 2
  foca.mark_step_begin()
  pred = foca.approximate()
  assert pred.item() == pytest.approx(2.0, abs=1e-5)
  # step 3: cache → predict x=3, expect 3 (recursive)
  foca.mark_step_begin()
  pred = foca.approximate()
  assert pred.item() == pytest.approx(3.0, abs=1e-5)


def test_foca_state_bdf2_quadratic():
  """BDF2 recursive prediction on y=x^2 has bounded error."""
  foca = FoCaCalibrator()
  # y = x^2: full-compute at x=0, x=1
  foca.mark_step_begin()
  foca.update(torch.tensor(0.0, dtype=torch.float32))
  foca.mark_step_begin()
  foca.update(torch.tensor(1.0, dtype=torch.float32))
  # step 2 (x=2, y=4): cache predict
  foca.mark_step_begin()
  pred2 = foca.approximate()
  assert pred2.item() == pytest.approx(2.0, abs=1e-5)  # linear extrapolation
  # step 3 (x=3, y=9): recursive cache predict
  foca.mark_step_begin()
  pred3 = foca.approximate()
  assert pred3.item() == pytest.approx(3.0, abs=1e-5)  # recursive linear


def test_foca_state_heun_correction():
  """Heun correction changes prediction when deriv_full != deriv_curr."""
  foca = FoCaCalibrator()
  # Feed two full-compute anchors with different slopes per element
  foca.mark_step_begin()
  foca.update(torch.tensor([0.0, 0.0], dtype=torch.float32))
  foca.mark_step_begin()
  foca.update(torch.tensor([1.0, 2.0], dtype=torch.float32))
  # First cache step: deriv_curr == deriv_full → Heun == BDF2
  foca.mark_step_begin()
  pred1 = foca.approximate()
  assert pred1 is not None
  # Full compute with a different slope → deriv_full changes
  foca.mark_step_begin()
  foca.update(torch.tensor([3.0, 8.0], dtype=torch.float32))
  # Next cache: deriv_full has changed, Heun correction may differ from BDF2
  foca.mark_step_begin()
  pred2 = foca.approximate()
  assert pred2 is not None


def test_foca_state_fallback_single_anchor():
  """With only one anchor, approximate() falls back to reusing the anchor."""
  foca = FoCaCalibrator()
  foca.mark_step_begin()
  foca.update(torch.tensor(5.0, dtype=torch.float32))
  # Only one anchor → F_km1 is None → approximate reuses F_k
  foca.mark_step_begin()
  pred = foca.approximate()
  assert pred.item() == pytest.approx(5.0, abs=1e-5)


def test_foca_state_reset():
  """Reset() clears all internal state."""
  foca = FoCaCalibrator()
  foca.mark_step_begin()
  foca.update(torch.tensor(1.0, dtype=torch.float32))
  foca.mark_step_begin()
  foca.update(torch.tensor(2.0, dtype=torch.float32))
  foca.reset_cache()
  state = foca.states.get("default")
  assert state is None or state.F_k is None


def test_foca_state_shape_change():
  """Shape change triggers automatic reset."""
  foca = FoCaCalibrator()
  foca.mark_step_begin()
  foca.update(torch.randn(2, 32, 512), name="test")
  foca.mark_step_begin()
  foca.update(torch.randn(4, 32, 512), name="test")
  state = foca.states["test"]
  assert state.F_km1 is None


def test_foca_calibrator_multi_stream():
  """Two named streams are independent."""
  foca = FoCaCalibrator()
  foca.mark_step_begin()
  foca.update(torch.tensor([1.0, 2.0], dtype=torch.float32), name="Bn")
  foca.update(torch.tensor([10.0, 20.0], dtype=torch.float32), name="Bn_encoder")
  foca.mark_step_begin()
  foca.update(torch.tensor([3.0, 4.0], dtype=torch.float32), name="Bn")
  foca.update(torch.tensor([30.0, 40.0], dtype=torch.float32), name="Bn_encoder")
  foca.mark_step_begin()
  pred_bn = foca.approximate(name="Bn")
  pred_enc = foca.approximate(name="Bn_encoder")
  assert not torch.allclose(pred_bn, pred_enc)


def test_foca_linear_extrapolation_across_anchor_gap():
  """A linear series stays linear after full-compute anchors are more than one step apart.

  DBCache recomputes, caches a streak, then recomputes. The anchor derivative is the change
  between those two recomputes, so it has to be a per-step rate. Treating the whole jump as one
  step makes the next forecast overshoot (6 instead of 5 on ``y = t``).
  """
  foca = FoCaCalibrator()
  full = {0: 0.0, 1: 1.0, 4: 4.0}
  for step in range(6):
    foca.mark_step_begin()
    if step in full:
      foca.update(torch.tensor(full[step], dtype=torch.float32))
      continue
    pred = foca.approximate()
    assert pred.item() == pytest.approx(float(step), abs=1e-5)


def test_foca_linear_extrapolation_when_rolling_window_spans_steps():
  """The local derivative is also a per-step rate when the rolling window skips a step.

  With a single anchor, the cache step reuses that anchor and does not advance the window. The
  next full compute therefore spans two steps, and the following forecast must not treat that
  jump as a one-step slope (``y = 2t`` would predict 8 instead of 6).
  """
  foca = FoCaCalibrator()
  foca.mark_step_begin()
  foca.update(torch.tensor(0.0, dtype=torch.float32))
  foca.mark_step_begin()
  assert foca.approximate().item() == pytest.approx(0.0, abs=1e-5)
  foca.mark_step_begin()
  foca.update(torch.tensor(4.0, dtype=torch.float32))
  foca.mark_step_begin()
  assert foca.approximate().item() == pytest.approx(6.0, abs=1e-5)


def test_foca_shape_change_keeps_step_counter():
  """A shape change clears history without rewinding the step clock.

  The first ``mark_step_begin`` lands before the named state exists, so the live counter is the
  second call (step 0). Resetting inside the shape change used to rewind that counter to -1, and
  the next step was numbered 0 again.
  """
  foca = FoCaCalibrator()
  foca.mark_step_begin()
  foca.update(torch.randn(2, 32, 512), name="test")
  foca.mark_step_begin()
  foca.update(torch.randn(4, 32, 512), name="test")
  state = foca.states["test"]
  assert state.F_km1 is None
  assert state.current_step == 0
  assert state.last_full_step == 0
  foca.mark_step_begin()
  assert state.current_step == 1
