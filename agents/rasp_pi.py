from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
from langchain.memory import ConversationBufferMemory
from langchain_community.cache import InMemoryCache

from langchain_community.llms.openai import OpenAI
from langchain_core.tools import ToolException


def _handle_error(error: ToolException) -> str:
    return (
            "The following errors occurred during tool execution:"
            + error.args[0]
            + "Please try another tool."
    )


cache = InMemoryCache()

manager_llm = OpenAI(
    model_name="gpt-4o-2024-05-13",
    temperature=0.5,
    max_tokens=8192,  # -1 returns as many tokens as possible given the prompt and the models maximal context size.
    top_p=1.0,  # Total probability mass of tokens to consider at each step.
    streaming=True,
    frequency_penalty=0,  # Penalizes repeated tokens according to frequency.
    presence_penalty=0,  # Penalizes repeated tokens.
    n=1,  # How many completions to generate for each prompt.
    best_of=1,  # Generates best_of completions server-side and returns the "best".
    max_retries=2,  # Maximum number of retries to make when generating.
    cache=cache,
)
agent_llm = OpenAI(
    model_name="gpt-4o",
    temperature=0.3,
    max_tokens=8192,  # -1 returns as many tokens as possible given the prompt and the models maximal context size.
    top_p=1.0,  # Total probability mass of tokens to consider at each step.
    streaming=True,
    frequency_penalty=0,  # Penalizes repeated tokens according to frequency.
    presence_penalty=0,  # Penalizes repeated tokens.
    n=1,  # How many completions to generate for each prompt.
    best_of=1,  # Generates best_of completions server-side and returns the "best".
    max_retries=2,  # Maximum number of retries to make when generating.
    cache=cache,
)

web_site_scrape_tool = ScrapeWebsiteTool()

shared_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


hw_expert = Agent(
    llm=agent_llm,
    role="Raspberry Pi Zero W hardware expert",
    goal=(
        """
        ### Instruction ###
        Scrape the web page and provide documentation details in {topic}
        """
    ),
    verbose=True,
    memory=False,
    tools=[web_site_scrape_tool],
    allow_delegation=False,
    backstory=(
        "As an AI expert on the Raspberry Pi Zero W, recommend the best way to connect and configure multiple sensors"
        " (e.g., accelerometer, temperature, light) simultaneously."
        " Include details on managing multiple I2C or SPI devices, power considerations,"
        " and any necessary software libraries."
    ),
    max_rpm=3
)

web_developer = Agent(
    llm=agent_llm,
    role="Python web application developer",
    goal=(
        """
        ### Instruction ###
        Develop Python based web application regarding {topic}
        """
    ),
    verbose=True,
    memory=False,
    tools=[web_site_scrape_tool],
    allow_delegation=False,
    backstory=(
        "You are a highly capable Python developer assistant."
        " Your role is to help me with any Python-related tasks, from coding to debugging to architectural design."
        " You have extensive knowledge of Python syntax, libraries, frameworks, and best practices."
        " You will provide clear, concise, and well-explained responses to my questions."
        " You will also proactively suggest improvements or alternative approaches to my code."
        " Your goal is to help me become a better Python programmer."
    )
)

refactor_expert = Agent(
    llm=agent_llm,
    role="Refactoring expert",
    goal=(
        """
        ### Instruction ###
        Review Python code and suggest ways to improve its readability, efficiency, and maintainability.
        """
    ),
    verbose=True,
    memory=False,
    tools=[web_site_scrape_tool],
    allow_delegation=False,
    backstory=(
        "You are an AI-powered Python refactoring expert."
        " Your task is to review my Python code and suggest ways to improve"
        " its readability, efficiency, and maintainability."
        " You have a deep understanding of Python coding conventions, design patterns"
        " and performance optimization techniques."
        " You will provide detailed explanations for your recommendations"
        " and work collaboratively with me to implement the changes."
        " Your goal is to help me write cleaner, more robust Python code."
        " You have an option to scrape web pages related to Python code concepts, coding conventions"
        " and design patterns in order to provide detailed explanations for your recommendations."
    )
)

sw_tester = Agent(
    llm=agent_llm,
    role="Expert in software testing using pytest",
    goal=(
        """
        ### Instruction ###
        Guide through a test-driven development (TDD) workflow for implementing a new feature in a Python project.
        """
    ),
    verbose=True,
    memory=False,
    tools=[web_site_scrape_tool],
    allow_delegation=False,
    backstory=(
        """
        You are an expert in software testing using pytest.
        I would like you to guide me through a test-driven development (TDD) workflow for implementing a new feature in a Python project.
        Please provide the following:
        - Write a high-level description of the feature you will implement.
        - Create a set of unit tests using pytest that define the expected behavior of the feature.
        - Run the tests, which should fail initially since the feature is not implemented yet.
        - Implement the feature in the production code.
        - Run the tests again, which should now pass.
        - Refactor the code if necessary to improve its quality and maintainability.
        - Provide the complete test code and production code for the feature.
        - Scrape web pages related to Python code concepts, coding conventions and design patterns to improve code quality."
        """
    )
)

hw_expert_task = Task(
    description="""
    ### Instruction ###
    Provide hardware description of connecting external devices regarding {topic}.
    """,
    expected_output="Short description of hardware connections.",
    tools=[web_site_scrape_tool],
    agent=hw_expert,
    async_execution=False,
)

web_developer_task = Task(
    description=(
        """
        ### Instruction ###
        Generate Python code for {topic}.
        Focus on getting data from hardware and displaying graph on web page.
        """
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
        """
        ### Instruction ###
        Refactor Python code for {topic}.
        
        ### Backstory ###
        You are an expert Python programmer and code reviewer.
        Your task is to review the provided Python code and suggest improvements.
        Provide a detailed analysis of the code, including:
        - Identifying any bugs, errors or potential issues
        - Suggesting refactoring opportunities to improve code readability, maintainability and efficiency
        - Pointing out any code smells or anti-patterns
        - Recommending best practices that should be followed
        Provide your review in a clear, structured format with specific examples and explanations.
        Feel free to include code snippets to illustrate your points.
        You have an option to scrape web pages related to Python code concepts, coding conventions
        and design patterns in order to provide a detailed analysis of the code.        
        """
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
        """
        ### Instruction ###
        Generate Python code for {topic}.
        Focus on pytest unit and functional tests using Selenium.
        """
    ),
    expected_output="Generate pytest code regarding {topic}.",
    tools=[web_site_scrape_tool],
    agent=sw_tester,
    async_execution=False,
    context=[hw_expert_task, web_developer_task, refactor_task],
)

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

result = crew.kickoff(
    inputs={
        "topic": (
            "Web application based on modern, dynamic and user friendly design patterns"
            " displaying daily, weekly, monthly or yearly historical data"
            " collected using Raspberry Pi Zero W 1-wire temperature sensor"
            " which are stored in Postgres database."
        )
    }
)

print(result)
