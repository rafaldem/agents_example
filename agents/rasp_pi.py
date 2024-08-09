from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, SeleniumScrapingTool

from langchain_openai import ChatOpenAI

from setup import *


# Create LLM agents
agent_llm = ChatOpenAI(
    model_name=AGENT_MODEL_NAME,
    temperature=AGENT_TEMPERATURE,
    streaming=AGENT_STREAMING,
    max_retries=AGENT_MAX_RETRIES,
)
manager_llm = ChatOpenAI(
    model_name=MANAGER_MODEL_NAME,
    temperature=MANAGER_TEMPERATURE,
    streaming=MANAGER_STREAMING,
    max_retries=MANAGER_MAX_RETRIES,
)

# Tools
selenium_tool = SeleniumScrapingTool(
    website_url=TOPIC_WEBPAGE_URL,
    wait_time=SELENIUM_SCRAPER_WAIT_TIME,
)


serper_search_tool = SerperDevTool()


# Agents
hw_expert = Agent(
    llm=agent_llm,
    role=HW_AGENT_ROLE,
    goal=HW_AGENT_GOAL,
    verbose=True,
    memory=False,
    tools=[serper_search_tool],
    allow_delegation=True,
    backstory=HW_AGENT_BACKSTORY,
    max_rpm=5,
)

web_developer = Agent(
    llm=agent_llm,
    role=WEB_DEV_ROLE,
    goal=WEB_DEV_GOAL,
    verbose=True,
    memory=False,
    tools=[serper_search_tool],
    allow_delegation=True,
    backstory=WEB_DEV_BACKSTORY
)

refactor_expert = Agent(
    llm=agent_llm,
    role=REFACTOR_EXPERT_ROLE,
    goal=REFACTOR_EXPERT_GOAL,
    verbose=True,
    memory=False,
    tools=[serper_search_tool],
    allow_delegation=True,
    backstory=REFACTOR_EXPERT_BACKSTORY
)

sw_tester = Agent(
    llm=agent_llm,
    role=SW_TESTER_ROLE,
    goal=SW_TESTER_GOAL,
    verbose=True,
    memory=False,
    tools=[serper_search_tool],
    allow_delegation=True,
    backstory=SW_TESTER_BACKSTORY
)

# Tasks
hw_expert_task = Task(
    description=HW_EXPERT_TASK_DESCRIPTION,
    expected_output=HW_EXPERT_TASK_EXPECTED_OUTPUT,
    tools=[serper_search_tool],
    agent=hw_expert,
    async_execution=False,
)

web_developer_task = Task(
    description=WEB_DEVELOPER_TASK_DESCRIPTION,
    expected_output=WEB_DEVELOPER_TASK_EXPECTED_OUTPUT,
    tools=[serper_search_tool],
    agent=web_developer,
    async_execution=False,
    context=[hw_expert_task],
)

refactor_task = Task(
    description=REFACTOR_TASK_DESCRIPTION,
    expected_output=REFACTOR_TASK_EXPECTED_OUTPUT,
    tools=[serper_search_tool],
    agent=web_developer,
    async_execution=False,
    context=[hw_expert_task, web_developer_task],
)

sw_tester_task = Task(
    description=SW_TESTER_TASK_DESCRIPTION,
    expected_output=SW_TESTER_TASK_EXPECTED_OUTPUT,
    tools=[serper_search_tool],
    agent=sw_tester,
    async_execution=False,
    context=[hw_expert_task, web_developer_task, refactor_task],
)

crew = Crew(
    agents=[hw_expert, web_developer, refactor_expert, sw_tester],
    tasks=[hw_expert_task, web_developer_task, refactor_task, sw_tester_task],
    process=Process.sequential,
    memory=False,
    cache=False,
    max_rpm=5,
    share_crew=True,
    full_output=True,
    verbose=2,
    manager_llm=manager_llm,
    output_file=CREW_OUTPUT_FILE,
)

# Execution
result = crew.kickoff(inputs={"topic": TOPIC})

print(result)
