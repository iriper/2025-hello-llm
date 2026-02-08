"""
Starter for demonstration of laboratory work.
"""

from pathlib import Path

# pylint: disable=too-many-locals, undefined-variable, unused-import


from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")
    
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analyzed_dataset = preprocessor.analyze()

    result = analyzed_dataset
    assert result is not None, "Demo does not work correctly"
    

if __name__ == "__main__":
    main()
