from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
from langchain.memory import ConversationBufferMemory
from langchain_community.cache import InMemoryCache
from langchain_community.llms.openai import OpenAI
from langchain_core.tools import ToolException


# Configuration constants

CACHE = InMemoryCache()

# Agent config
# not default
AGENT_MODEL_NAME = "gpt-4o"
AGENT_TEMPERATURE = 0.3
AGENT_MAX_TOKENS = 8192
AGENT_STREAMING = True
# default
AGENT_TOP_P = 1.0
AGENT_FREQUENCY_PENALTY = 0
AGENT_PRESENCE_PENALTY = 0
AGENT_N = 1
AGENT_BEST_OF = 1
AGENT_MAX_RETRIES = 2

# Manager config
# not default
MANAGER_MODEL_NAME = "gpt-4o-2024-05-13"
MANAGER_TEMPERATURE = 0.5
MANAGER_MAX_TOKENS = 8192
MANAGER_STREAMING = True
# default
MANAGER_TOP_P = 1.0
MANAGER_FREQUENCY_PENALTY = 0
MANAGER_PRESENCE_PENALTY = 0
MANAGER_N = 1
MANAGER_BEST_OF = 1
MANAGER_MAX_RETRIES = 2


# Handle errors during tool execution
def handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution: "
        + error.args[0]
        + " Please try another tool."
    )


# Create LLM agents
agent_llm = OpenAI(
    model_name=AGENT_MODEL_NAME,
    temperature=AGENT_TEMPERATURE,
    max_tokens=AGENT_MAX_TOKENS,
    streaming=AGENT_STREAMING,
)
manager_llm = OpenAI(
    model_name=MANAGER_MODEL_NAME,
    temperature=MANAGER_TEMPERATURE,
    max_tokens=MANAGER_MAX_TOKENS,
    streaming=MANAGER_STREAMING,
)


# Tools and memory
web_site_scrape_tool = ScrapeWebsiteTool()
shared_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Agents
hw_expert = Agent(
    llm=agent_llm,
    role="Raspberry Pi Zero W hardware expert",
    goal=(
        "### Instruction ###\n"
        "Scrape the web page and provide documentation details in {topic}\n"
    ),
    verbose=True,
    memory=True,
    tools=[web_site_scrape_tool],
    allow_delegation=False,
    backstory=(
        "As an AI expert on the Raspberry Pi Zero W, recommend the best way to connect and configure multiple sensors"
        " (e.g., accelerometer, temperature, light) simultaneously.\n"
        "Include details on managing multiple I2C or SPI devices, power considerations,"
        " and any necessary software libraries.\n"
    ),
    max_rpm=3
)

web_developer = Agent(
    llm=agent_llm,
    role="Python web application developer",
    goal=(
        "### Instruction ###\n"
        "Develop Python based web application regarding {topic}\n"
    ),
    verbose=True,
    memory=True,
    tools=[web_site_scrape_tool],
    allow_delegation=False,
    backstory=(
        "You are a highly capable Python developer assistant.\n"
        "Your role is to help me with any Python-related tasks, from coding to debugging to architectural design.\n"
        "You have extensive knowledge of Python syntax, libraries, frameworks, and best practices.\n"
        "You will provide clear, concise, and well-explained responses to my questions.\n"
        "You will also proactively suggest improvements or alternative approaches to my code.\n"
        "Your goal is to help me become a better Python programmer.\n"
    )
)

refactor_expert = Agent(
    llm=agent_llm,
    role="Refactoring expert",
    goal=(
        "### Instruction ###\n"
        "Review Python code and suggest ways to improve its readability, efficiency, and maintainability.\n"
    ),
    verbose=True,
    memory=False,
    tools=[web_site_scrape_tool],
    allow_delegation=False,
    backstory=(
        "You are an AI-powered Python refactoring expert.\n"
        "Your task is to review my Python code and suggest improvements.\n"
        "You have a deep understanding of Python coding conventions, design patterns,"
        " and performance optimization techniques."
        "You will provide detailed explanations for your recommendations"
        " and work collaboratively with me to implement the changes.\n"
        "Your goal is to help me write cleaner, more robust Python code.\n"
        "You have an option to scrape web pages related to Python code concepts, coding conventions,"
        " and design patterns in order to provide detailed explanations for your recommendations.\n"
    )
)

sw_tester = Agent(
    llm=agent_llm,
    role="Expert in software testing using pytest",
    goal=(
        "### Instruction ###\n"
        "Guide through a test-driven development (TDD) workflow for implementing a new feature in a Python project.\n"
    ),
    verbose=True,
    memory=False,
    tools=[web_site_scrape_tool],
    allow_delegation=False,
    backstory=(
        "You are an expert in software testing using pytest.\n"
        "I would like you to guide me through a test-driven development (TDD) workflow"
        " for implementing a new feature in a Python project."
        "Please provide the following:\n"
        "- Write a high-level description of the feature you will implement.\n"
        "- Create a set of unit tests using pytest that define the expected behavior of the feature.\n"
        "- Run the tests, which should fail initially since the feature is not implemented yet.\n"
        "- Implement the feature in the production code.\n"
        "- Run the tests again, which should now pass.\n"
        "- Refactor the code if necessary to improve its quality and maintainability.\n"
        "- Provide the complete test code and production code for the feature.\n"
        "- Scrape web pages related to Python code concepts,"
        " coding conventions and design patterns to improve code quality.\n"
    )
)

# Tasks
hw_expert_task = Task(
    description=(
        "### Instruction ###\n"
        "Provide hardware description of connecting external devices regarding {topic}.\n"
    ),
    expected_output="Short description of hardware connections.",
    tools=[web_site_scrape_tool],
    agent=hw_expert,
    async_execution=False,
)

web_developer_task = Task(
    description=(
        "### Instruction ###\n"
        "Generate Python code for {topic}.\n"
        "Focus on getting data from hardware and displaying graph on web page.\n"
    ),
    expected_output=(
        "Generate classes on {topic} that could be used to eventually expand functionality with "
        "additional content."
    ),
    tools=[web_site_scrape_tool],
    agent=web_developer,
    async_execution=False,
    context=[hw_expert_task],
)

refactor_task = Task(
    description=(
        "### Instruction ###\n"
        "Refactor Python code for {topic}.\n"
        "\n"
        "### Backstory ###\n"
        "You are an expert Python programmer and code reviewer.\n"
        "Your task is to review the provided Python code and suggest improvements.\n"
        "Provide a detailed analysis of the code, including:\n"
        "- Identifying any bugs, errors or potential issues\n"
        "- Suggesting refactoring opportunities to improve code readability, maintainability and efficiency\n"
        "- Pointing out any code smells or anti-patterns\n"
        "- Recommending best practices that should be followed\n"
        "Provide your review in a clear, structured format with specific examples and explanations.\n"
        "Feel free to include code snippets to illustrate your points.\n"
        "You have an option to scrape web pages related to Python code concepts, coding conventions"
        " and design patterns in order to provide a detailed analysis of the code.\n"
    ),
    expected_output=(
        "Generate classes on {topic} that could be used to eventually expand functionality with "
        "additional content."
    ),
    tools=[web_site_scrape_tool],
    agent=web_developer,
    async_execution=False,
    context=[hw_expert_task, web_developer_task],
)

sw_tester_task = Task(
    description=(
        "### Instruction ###\n"
        "Generate Python code for {topic}.\n"
        "Focus on pytest unit and functional tests using Selenium.\n"
    ),
    expected_output="Generate pytest code regarding {topic}.",
    tools=[web_site_scrape_tool],
    agent=sw_tester,
    async_execution=False,
    context=[hw_expert_task, web_developer_task, refactor_task],
)

# Crew
crew = Crew(
    agents=[hw_expert, web_developer, sw_tester],
    tasks=[hw_expert_task, web_developer_task, refactor_task, sw_tester_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=3,
    share_crew=True,
    full_output=True,
    verbose=2,
    manager_llm=manager_llm,
)

# Execution
result = crew.kickoff(
    inputs={
        "topic": (
            "Web application based on modern, dynamic and user-friendly design patterns"
            " displaying daily, weekly, monthly or yearly historical data"
            " collected using Raspberry Pi Zero W 1-wire temperature sensor"
            " which are stored in Postgres database."
        )
    }
)

print(result)
