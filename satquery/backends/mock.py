from satquery.backends.base import OpticalPrediction
from satquery.schemas import Box, LoadedImage


class MockBackend:
    name = "mock/development-v1"

    def analyze(self, image: LoadedImage, question: str) -> OpticalPrediction:
        h, w = image.rgb.shape[:2]
        return OpticalPrediction(
            answer="MOCK / DEVELOPMENT MODE: the box marks a synthetic demonstration region. "
            "No satellite feature was identified and your question has not been answered by a model.",
            boxes=[
                Box(x1=w * 0.25, y1=h * 0.25, x2=w * 0.65, y2=h * 0.65, label="DEMO REGION — no inference")
            ],
            backend=self.name,
            demonstration=True,
            warnings=["Synthetic overlay for testing only. Select Qwen on suitable hardware for optical QA."],
            provenance={"inference_performed": False},
        )
