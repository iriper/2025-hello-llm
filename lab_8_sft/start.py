"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements


from pathlib import Path
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
