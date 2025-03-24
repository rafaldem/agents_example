import logging
import asyncio

from setup import *

from together import AsyncTogether, Together
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMRunner:
    """Manages interaction with LLMs for both individual models and aggregators."""

    def __init__(
            self,
            user_prompt: str,
            reference_models: List[str],
            aggregator_model: str,
            aggregator_prompt: str
    ):
        self.user_prompt = user_prompt
        self.reference_models = reference_models
        self.aggregator_model = aggregator_model
        self.aggregator_prompt = aggregator_prompt

        # Initialize clients
        self.client = Together()
        self.async_client = AsyncTogether()

    async def run_single_model(self, model: str) -> Optional[str]:
        """Run a single model asynchronously."""
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": self.user_prompt}],
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            return None

    async def gather_responses(self) -> List[Optional[str]]:
        """Run all reference models concurrently."""
        tasks = [self.run_single_model(model) for model in self.reference_models]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def run_aggregator(self, responses: List[str]) -> None:
        """Aggregate responses synchronously using the aggregator model."""
        try:
            final_stream = self.client.chat.completions.create(
                model=self.aggregator_model,
                messages=[
                    {"role": "system", "content": self.aggregator_prompt},
                    {"role": "user", "content": ",".join(filter(None, responses))},
                ],
                stream=True,
            )
            for chunk in final_stream:
                logger.info(chunk.choices[0].delta.content or "")
        except Exception as e:
            logger.error(f"Error with aggregator: {e}")

    async def execute(self) -> None:
        """Main execution pipeline."""
        responses = await self.gather_responses()
        filtered_responses = [resp for resp in responses if resp is not None]
        if not filtered_responses:
            print("No valid responses to aggregate.")
            return
        self.run_aggregator(filtered_responses)


if __name__ == "__main__":

    # Execute
    runner = LLMRunner(
        user_prompt=TOPIC,
        reference_models=REFERENCE_MODELS,
        aggregator_model=AGGREGATOR_MODEL,
        aggregator_prompt=AGGREGATOR_PROMPT,
    )
    asyncio.run(runner.execute())
