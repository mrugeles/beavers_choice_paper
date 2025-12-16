import os
from smolagents import ToolCallingAgent, OpenAIServerModel
from quote_agent.quote_agent import QuoteAgent
from inventory_agent.inventory_agent import InventoryAgent
import dotenv
dotenv.load_dotenv()

openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

quote_agent = QuoteAgent(model=model)
inventory_agent = InventoryAgent(model=model)


class OrchestratorAgent(ToolCallingAgent):

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            model=model,
            name="orchestrator_agent",
            managed_agents=[quote_agent, inventory_agent],
            description="Orchestrates tasks by routing requests to the appropriate agent, either for handling quotes or for managing inventory",
        )
