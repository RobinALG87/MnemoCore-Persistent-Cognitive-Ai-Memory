"""
Tests for Phase 5.0 Prediction Store Module.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta

from mnemocore.core.prediction_store import (
    PredictionRecord,
    PredictionStore,
    STATUS_PENDING,
    STATUS_VERIFIED,
    STATUS_FALSIFIED,
    STATUS_EXPIRED,
)


# ------------------------------------------------------------------ #
#  PredictionRecord                                                  #
# ------------------------------------------------------------------ #

class TestPredictionRecord:
    def test_auto_id(self):
        r = PredictionRecord(content="AI will achieve AGI by 2030")
        assert r.id.startswith("pred_")

    def test_default_status_pending(self):
        r = PredictionRecord()
        assert r.status == STATUS_PENDING

    def test_is_expired_no_deadline(self):
        r = PredictionRecord()
        assert not r.is_expired()

    def test_is_expired_future_deadline(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        r = PredictionRecord(verification_deadline=future)
        assert not r.is_expired()

    def test_is_expired_past_deadline(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        r = PredictionRecord(verification_deadline=past, status=STATUS_PENDING)
        assert r.is_expired()

    def test_not_expired_if_not_pending(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        r = PredictionRecord(verification_deadline=past, status=STATUS_VERIFIED)
        assert not r.is_expired()

    def test_to_dict_keys(self):
        r = PredictionRecord(content="test")
        d = r.to_dict()
        for key in ["id", "content", "status", "confidence_at_creation", "outcome"]:
            assert key in d

    def test_roundtrip(self):
        r = PredictionRecord(content="X will happen", confidence_at_creation=0.75)
        restored = PredictionRecord.from_dict(r.to_dict())
        assert restored.content == r.content
        assert restored.confidence_at_creation == pytest.approx(0.75)
        assert restored.status == STATUS_PENDING


# ------------------------------------------------------------------ #
#  PredictionStore                                                   #
# ------------------------------------------------------------------ #

class TestPredictionStore:
    def test_create_returns_id(self):
        store = PredictionStore()
        pred_id = store.create("Test prediction", confidence=0.8)
        assert pred_id.startswith("pred_")

    def test_get_returns_record(self):
        store = PredictionStore()
        pred_id = store.create("Something will happen", confidence=0.6)
        rec = store.get(pred_id)
        assert rec is not None
        assert rec.content == "Something will happen"

    def test_get_unknown_returns_none(self):
        store = PredictionStore()
        assert store.get("pred_nonexistent") is None

    def test_list_all_no_filter(self):
        store = PredictionStore()
        store.create("A", confidence=0.5)
        store.create("B", confidence=0.7)
        all_preds = store.list_all()
        assert len(all_preds) == 2

    def test_list_filter_by_status(self):
        store = PredictionStore()
        store.create("A", confidence=0.5)
        pending = store.list_all(status=STATUS_PENDING)
        assert len(pending) == 1

    def test_len(self):
        store = PredictionStore()
        store.create("X")
        store.create("Y")
        assert len(store) == 2

    def test_deadline_days(self):
        store = PredictionStore()
        pred_id = store.create("future", deadline_days=30)
        rec = store.get(pred_id)
        assert rec.verification_deadline is not None
        deadline = datetime.fromisoformat(rec.verification_deadline)
        # Should be ~30 days from now
        days_diff = (deadline - datetime.now(timezone.utc)).days
        assert 28 <= days_diff <= 31

    @pytest.mark.asyncio
    async def test_verify_success(self):
        store = PredictionStore(engine=None)
        pred_id = store.create("Test will succeed", confidence=0.9)
        result = await store.verify(pred_id, success=True, notes="confirmed")
        assert result is not None
        assert result.status == STATUS_VERIFIED
        assert result.outcome is True
        assert result.verification_notes == "confirmed"

    @pytest.mark.asyncio
    async def test_verify_failure(self):
        store = PredictionStore(engine=None)
        pred_id = store.create("This will fail", confidence=0.9)
        result = await store.verify(pred_id, success=False)
        assert result.status == STATUS_FALSIFIED
        assert result.outcome is False

    @pytest.mark.asyncio
    async def test_verify_unknown_returns_none(self):
        store = PredictionStore()
        result = await store.verify("pred_bogus", success=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_expire_due(self):
        store = PredictionStore()
        # Past deadline
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        rec = PredictionRecord(content="overdue", verification_deadline=past)
        store._records[rec.id] = rec
        expired = await store.expire_due()
        assert len(expired) == 1
        assert expired[0].status == STATUS_EXPIRED

    def test_get_due(self):
        store = PredictionStore()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        rec_past = PredictionRecord(content="overdue", verification_deadline=past)
        rec_future = PredictionRecord(content="future", verification_deadline=future)
        store._records[rec_past.id] = rec_past
        store._records[rec_future.id] = rec_future
        due = store.get_due()
        assert rec_past in due
        assert rec_future not in due
