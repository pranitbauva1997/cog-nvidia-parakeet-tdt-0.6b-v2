from cog import BasePredictor, Input, Path
import nemo.collections.asr as nemo_asr
from typing import Any, Dict, List, Union

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the Parakeet-TDT-0.6b-v2 ASR model into memory."""
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

    def predict(
        self,
        audio: Path = Input(description="16kHz mono .wav or .flac audio file to transcribe."),
        timestamps: bool = Input(description="Return word/segment/char timestamps in output.", default=False),
    ) -> Union[str, Dict[str, Any]]:
        """Transcribe the input audio file. Optionally return timestamps."""
        output = self.asr_model.transcribe([str(audio)], timestamps=timestamps)
        result = output[0]
        if timestamps:
            # Return both text and all timestamp levels
            return {
                "text": result.text,
                "timestamps": result.timestamp
            }
        else:
            return result.text
