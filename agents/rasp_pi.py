from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool, TXTSearchTool


web_site_search_tool = WebsiteSearchTool()
text_search_tool = TXTSearchTool()


hw_expert = Agent(
  role='Raspberry Pi Zero W hardware expert',
  goal='Provide documentation details in {topic}',
  verbose=True,
  memory=True,
  tools=[web_site_search_tool],
  allow_delegation=True
)


web_developer = Agent(
  role='Web application developer',
  goal='Develop Python based web application',
  verbose=True,
  memory=True,
  tools=[text_search_tool],
  allow_delegation=True
)


sw_tester = Agent(
  role='Software tester',
  goal='Developing unit and functional tests',
  verbose=True,
  memory=True,
  tools=[text_search_tool],
  allow_delegation=False
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
  tools=[text_search_tool],
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


result = crew.kickoff(inputs={'topic': 'Raspberry Pi Zero W 1-wire temperature sensor'})
print(result)
