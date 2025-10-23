from spec_repair.wrappers.spec import Spec
from multiprocessing import Lock


class ConcurrentSpecRecorder:
    def __init__(self):
        self.storage: dict[Spec, int] = dict()
        self._lock = Lock()

    def add(self, new_spec: Spec):
        with self._lock:
            for spec, index in self.storage.items():
                if spec == new_spec:
                    return index
            index = len(self.storage)
            self.storage[new_spec] = index
            return index

    def get_id(self, new_spec: Spec):
        with self._lock:
            for spec, index in self.storage.values():
                if spec == new_spec:
                    return index
            return -1

    def __str__(self):
        return str(self.storage)

    def get_storage(self) -> dict[Spec, int]:
        return self.storage
