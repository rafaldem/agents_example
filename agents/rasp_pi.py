import asyncio

from setup import *

from together import AsyncTogether, Together


class LLMRunner:
    """Manages interaction with LLMs for both individual models and aggregators."""

    def __init__(self, user_prompt, reference_models, aggregator_model, aggregator_prompt):
        self.user_prompt = user_prompt
        self.reference_models = reference_models
        self.aggregator_model = aggregator_model
        self.aggregator_prompt = aggregator_prompt

        # Initialize clients
        self.client = Together()
        self.async_client = AsyncTogether()

    async def run_single_model(self, model):
        """Run a single model asynchronously."""
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": self.user_prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with model {model}: {e}")
            return None

    async def gather_responses(self):
        """Run all reference models concurrently."""
        tasks = [self.run_single_model(model) for model in self.reference_models]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def run_aggregator(self, responses):
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
                print(chunk.choices[0].delta.content or "", end="", flush=True)
        except Exception as e:
            print(f"Error with aggregator: {e}")

    async def execute(self):
        """Main execution pipeline."""
        responses = await self.gather_responses()
        filtered_responses = [resp for resp in responses if resp is not None]
        if not filtered_responses:
            print("No valid responses to aggregate.")
            return
        self.run_aggregator(filtered_responses)


if __name__ == "__main__":
    USER_PROMPT = TOPIC
    REFERENCE_MODELS = [
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen1.5-72B-Chat",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "databricks/dbrx-instruct",
    ]
    AGGREGATOR_MODEL = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    AGGREGATOR_PROMPT = """You have been provided with a set of responses from various open-source models...
                           Your task is to synthesize these responses into a single, high-quality response."""

    # Execute
    runner = LLMRunner(
        user_prompt=USER_PROMPT,
        reference_models=REFERENCE_MODELS,
        aggregator_model=AGGREGATOR_MODEL,
        aggregator_prompt=AGGREGATOR_PROMPT,
    )
    asyncio.run(runner.execute())
