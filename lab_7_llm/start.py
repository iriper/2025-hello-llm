"""
Starter for demonstration of laboratory work.
"""

from pathlib import Path

# pylint: disable=too-many-locals, undefined-variable, unused-import


from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset

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

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    batch_size = 1
    max_length = 120
    device = "cpu"
    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    model_analysis = pipeline.analyze_model()
    print(model_analysis)
    sample = dataset[0]
    sample_pred = pipeline.infer_sample(sample)
    print(sample_pred)
    

if __name__ == "__main__":
    main()
