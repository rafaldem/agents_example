from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPER_API_KEY"] = ""

llm = ChatOpenAI(
  model="crewai-llama2",
  base_url="http://localhost:11434/v1"
)

web_site_search_tool = SerperDevTool()


hw_expert = Agent(
  llm=llm,
  role='Raspberry Pi Zero W hardware expert',
  goal='Provide documentation details in {topic}',
  verbose=True,
  memory=True,
  tools=[web_site_search_tool],
  allow_delegation=True,
  backstory="You are a hardware expert that provides specific details based on documentation."
)

web_developer = Agent(
  llm=llm,
  role='Web application developer',
  goal='Develop Python based web application',
  verbose=True,
  memory=True,
  tools=[web_site_search_tool],
  allow_delegation=True,
  backstory="You are a web application developer that provides Python code."
)

sw_tester = Agent(
  llm=llm,
  role='Software tester',
  goal='Developing unit and functional tests.',
  verbose=True,
  memory=True,
  tools=[web_site_search_tool],
  allow_delegation=False,
  backstory="You are a software tester that provides unit and functional tests using Python code."
)

hw_expert_task = Task(
  description="Focus on hardware description of connecting external devices regarding {topic}.",
  expected_output='Short description of hardware connections.',
  tools=[web_site_search_tool],
  agent=hw_expert,
)

web_developer_task = Task(
  description=(
    "Generate Python code for {topic}."
    "Focus on getting data from hardware and displaying graph on web page."
  ),
  expected_output=(
    'Generate classes on {topic} that could be used to eventually expand functionality with '
    'additional content.'
  ),
  tools=[web_site_search_tool],
  agent=web_developer,
  async_execution=False,
)

sw_tester_task = Task(
  description=(
    "Generate Python code for {topic}."
    "Focus on pytest unit and functional tests using Selenium."
  ),
  expected_output='Generate pytest code regarding {topic}.',
  tools=[web_site_search_tool],
  agent=web_developer,
  async_execution=False,
)

crew = Crew(
  agents=[hw_expert, web_developer, sw_tester],
  tasks=[hw_expert_task, web_developer_task, sw_tester_task],
  process=Process.sequential,
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)


result = crew.kickoff(
  inputs={
    'topic': (
      'Web application displaying daily, weekly or monthly historical data of '
      'Raspberry Pi Zero W 1-wire temperature sensor'
    )
  }
)
print(result)
