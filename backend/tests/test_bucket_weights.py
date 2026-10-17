"""
backend/tests/test_bucket_weights.py

Identity proof for the Settings bucket-weight layer (weights-08 stream,
decision 0008 item 6).

- At default weights the stored value is returned exactly unchanged.
- Non-default weights move the value in the expected direction.
- The bucket map partitions ml.utils.FEATURE_COLS exactly, so a new
  feature column without a bucket decision fails loudly.
"""

from __future__ import annotations

import math

import pytest

from ml import bucket_weights as bw
from ml.utils import FEATURE_COLS


def _weights(**overrides: float) -> dict[str, float]:
    w = dict(bw.DEFAULT_BUCKET_WEIGHTS)
    w.update(overrides)
    return w


def _attributions(value: float = 0.0, **overrides: float) -> dict[str, float]:
    phi = {key: value for key in bw.BUCKET_KEYS}
    phi.update(overrides)
    return phi


class TestBucketMapPartition:
    def test_partitions_feature_cols_exactly(self):
        bw.validate_bucket_map(FEATURE_COLS)

    def test_counts(self):
        weighted = sum(len(m) for m in bw.BUCKET_MEMBERS.values())
        assert weighted == 54
        assert len(bw.UNWEIGHTED_MEMBERS) == 18
        assert weighted + len(bw.UNWEIGHTED_MEMBERS) == len(FEATURE_COLS) == 72

    def test_new_column_without_a_bucket_fails(self):
        with pytest.raises(ValueError, match="no group"):
            bw.validate_bucket_map([*FEATURE_COLS, "brand_new_feature"])

    def test_unknown_column_raises_key_error(self):
        with pytest.raises(KeyError):
            bw.bucket_of("not_a_feature")

    def test_every_feature_col_has_a_bucket(self):
        for col in FEATURE_COLS:
            assert bw.bucket_of(col) in (*bw.BUCKET_KEYS, bw.UNWEIGHTED_KEY)


class TestDefaults:
    def test_sum_to_100(self):
        assert math.fsum(bw.DEFAULT_BUCKET_WEIGHTS.values()) == pytest.approx(100.0)

    def test_every_default_positive(self):
        # r_g = w_g / d_g is undefined for d_g <= 0.
        for key in bw.BUCKET_KEYS:
            assert bw.DEFAULT_BUCKET_WEIGHTS[key] > 0

    def test_weights_are_default(self):
        assert bw.weights_are_default(dict(bw.DEFAULT_BUCKET_WEIGHTS)) is True
        assert bw.weights_are_default(_weights(kalman_form=36.0)) is False


class TestIdentityAtDefaults:
    def test_defaults_return_stored_value_unchanged(self):
        out = bw.apply_bucket_weights(
            17.5, _attributions(kalman_form=3.0), dict(bw.DEFAULT_BUCKET_WEIGHTS)
        )
        assert out.adjusted == 17.5
        assert out.weighting == "default"
        assert out.reason == "defaults"
        assert out.clipped is False

    def test_identity_holds_for_any_attributions(self):
        phi = _attributions(
            kalman_form=3.0,
            seasonal_baseline=-1.25,
            matchup=0.5,
            rule_meta=10.0,
        )
        out = bw.apply_bucket_weights(
            0.0, phi, dict(bw.DEFAULT_BUCKET_WEIGHTS)
        )
        assert out.adjusted == 0.0
        assert out.weighting == "default"


class TestDirection:
    def test_raising_positive_bucket_raises_projection(self):
        out = bw.apply_bucket_weights(
            20.0, _attributions(kalman_form=3.0), _weights(kalman_form=50.0)
        )
        assert out.adjusted == pytest.approx(20.0 + (50.0 / 35.0 - 1) * 3.0)
        assert out.adjusted > 20.0
        assert out.weighting == "user"
        assert out.reason is None

    def test_lowering_positive_bucket_lowers_projection(self):
        out = bw.apply_bucket_weights(
            20.0, _attributions(kalman_form=3.0), _weights(kalman_form=20.0)
        )
        assert out.adjusted < 20.0

    def test_zero_weight_removes_group_contribution_exactly(self):
        out = bw.apply_bucket_weights(
            20.0, _attributions(kalman_form=3.0), _weights(kalman_form=0.0)
        )
        assert out.adjusted == pytest.approx(20.0 - 3.0)
        assert out.weighting == "user"

    def test_raising_negative_bucket_lowers_projection(self):
        out = bw.apply_bucket_weights(
            20.0, _attributions(matchup=-2.0), _weights(matchup=30.0)
        )
        assert out.adjusted < 20.0

    def test_small_group_is_proportionally_more_sensitive(self):
        kalman = bw.apply_bucket_weights(
            0.0, _attributions(kalman_form=1.0), _weights(kalman_form=36.0)
        )
        rule = bw.apply_bucket_weights(
            0.0, _attributions(rule_meta=1.0), _weights(rule_meta=6.0)
        )
        assert rule.adjusted > kalman.adjusted > 0.0


class TestRatioCap:
    def test_ratio_capped_at_two(self):
        assert bw.bucket_ratio(100.0, 5.0) == pytest.approx(2.0)
        assert bw.bucket_ratio(10.0, 5.0) == pytest.approx(2.0)
        assert bw.bucket_ratio(0.0, 5.0) == pytest.approx(0.0)

    def test_cap_applies_in_formula(self):
        out = bw.apply_bucket_weights(
            10.0, _attributions(rule_meta=1.0), _weights(rule_meta=100.0)
        )
        assert out.adjusted == pytest.approx(10.0 + (2.0 - 1.0) * 1.0)


class TestMissingAttributions:
    def test_none_attributions_return_stored_with_reason(self):
        out = bw.apply_bucket_weights(17.5, None, _weights(kalman_form=50.0))
        assert out.adjusted == 17.5
        assert out.weighting == "default"
        assert out.reason == "no_attribution"

    def test_non_finite_attribution_returns_stored_with_reason(self):
        out = bw.apply_bucket_weights(
            17.5, _attributions(kalman_form=float("nan")), _weights(kalman_form=50.0)
        )
        assert out.adjusted == 17.5
        assert out.reason == "no_attribution"


class TestClipping:
    def test_negative_result_floors_at_zero(self):
        out = bw.apply_bucket_weights(
            1.0, _attributions(kalman_form=10.0), _weights(kalman_form=0.0)
        )
        assert out.adjusted == 0.0
        assert out.clipped is True

    def test_no_clip_when_allowed_negative(self):
        out = bw.apply_bucket_weights(
            1.0,
            _attributions(kalman_form=10.0),
            _weights(kalman_form=0.0),
            non_negative=False,
        )
        assert out.adjusted == pytest.approx(-9.0)
        assert out.clipped is False
