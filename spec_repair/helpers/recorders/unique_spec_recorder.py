from typing import Optional

from spec_repair.helpers.recorders.unique_recorder import UniqueRecorder
from spec_repair.helpers.spectra_specification import SpectraSpecification
from spec_repair.util.file_util import write_to_file
from spec_repair.wrappers.spec import Spec


class UniqueSpecRecorder(UniqueRecorder[SpectraSpecification]):
    def __init__(self, debug_folder: Optional[str] = None):
        super().__init__()
        self.debug_folder = debug_folder

    def add(self, new_spec: SpectraSpecification):
        index = super().add(new_spec)
        if self.debug_folder:
            write_to_file(f"{self.debug_folder}/spec_{index}.spectra", new_spec.to_str())
        return index

    def get_specs(self) -> list[str]:
        return [spec.to_str() for spec in self._value_to_id.keys()]
