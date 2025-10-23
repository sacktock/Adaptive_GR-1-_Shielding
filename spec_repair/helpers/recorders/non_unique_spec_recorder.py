from typing import Optional

from spec_repair.helpers.recorders.non_unique_recorder import NonUniqueRecorder
from spec_repair.util.file_util import write_to_file
from spec_repair.wrappers.spec import Spec


class NonUniqueSpecRecorder(NonUniqueRecorder[Spec]):
    def __init__(self, debug_folder: Optional[str] = None):
        super().__init__()
        self.debug_folder = debug_folder

    def add(self, new_spec: Spec):
        index = super().add(new_spec)
        if self.debug_folder:
            write_to_file(f"{self.debug_folder}/spec_{index}.spectra", new_spec.get_spec())
        return index

    def get_specs(self) -> list[str]:
        return [spec.get_spec() for spec in self._list]
