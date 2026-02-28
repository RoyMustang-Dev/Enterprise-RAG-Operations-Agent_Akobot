"""
Persistent Job Tracker

Stores ingestion/crawler job state in SQLite so multi-worker environments can track progress.
"""
from app.infra.database import upsert_ingestion_job


class _PersistentList(list):
    def __init__(self, on_change, *args):
        super().__init__(*args)
        self._on_change = on_change

    def append(self, item):
        super().append(item)
        self._on_change()

    def extend(self, items):
        super().extend(items)
        self._on_change()

    def __setitem__(self, idx, value):
        super().__setitem__(idx, value)
        self._on_change()

    def pop(self, *args, **kwargs):
        value = super().pop(*args, **kwargs)
        self._on_change()
        return value


class JobTracker(dict):
    def __init__(self, job_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._job_id = job_id
        self._persist()

    def _persist(self):
        status = self.get("status", "pending")
        upsert_ingestion_job(self._job_id, dict(self), status=status)

    def __setitem__(self, key, value):
        if key == "logs" and isinstance(value, list) and not isinstance(value, _PersistentList):
            value = _PersistentList(self._persist, value)
        super().__setitem__(key, value)
        self._persist()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._persist()

    def setdefault(self, key, default=None):
        if key == "logs" and default == []:
            default = _PersistentList(self._persist)
        value = super().setdefault(key, default)
        if key == "logs" and isinstance(value, list) and not isinstance(value, _PersistentList):
            value = _PersistentList(self._persist, value)
            super().__setitem__(key, value)
        self._persist()
        return value
