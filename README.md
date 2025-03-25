# Raspberry Pi Zero W Temperature Monitoring System

## Project Overview

This project implements an autonomous, AI-driven system for collecting, analyzing, and visualizing temperature data using a Raspberry Pi Zero W. The system leverages CrewAI framework to create a collaborative team of specialized AI agents that work together to develop a complete solution for temperature monitoring and visualization.

## Architecture

The system consists of the following components:

1. **Hardware**: Raspberry Pi Zero W with a 1-wire temperature sensor
2. **Database**: PostgreSQL database for storing historical temperature data
3. **Web Application**: Python-based web application for displaying temperature data with various time period visualizations
4. **AI Agents**: A crew of specialized AI agents built with CrewAI for different aspects of the solution

## AI Agent Crew

The project utilizes CrewAI to orchestrate a team of specialized AI agents:

1. **Hardware Expert Agent**: Provides expertise on connecting and configuring the Raspberry Pi Zero W with temperature sensors, focusing on hardware connections and configurations.

2. **Web Developer Agent**: Develops the Python-based web application for retrieving data from the hardware and displaying it on dynamic, user-friendly web pages.

3. **Refactoring Expert Agent**: Reviews and improves code quality, focusing on readability, efficiency, and maintainability.

4. **Software Testing Agent**: Implements comprehensive testing using pytest and follows test-driven development (TDD) methodology.

## Key Features

- Real-time temperature data collection from Raspberry Pi Zero W
- Historical data storage in PostgreSQL database
- Flexible data visualization (daily, weekly, monthly, yearly views)
- Modern, dynamic, and user-friendly web interface
- Comprehensive testing suite using pytest
- Collaborative AI agent system for ongoing development and improvement

## Prerequisites

- Raspberry Pi Zero W
- 1-wire temperature sensor
- PostgreSQL database
- Python 3.x
- OpenAI API key (for the AI agents)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rafaldem/agents_example
   cd agents_example
   ```

2. Set up the environment:
   ```
   pip install -r requirements.txt
   ```

3. Configure the OpenAI API key:
   - Create a `.env` file in the agents directory
   - Add your OpenAI API key: `OPENAI_API_KEY = "your-api-key"`

## Usage

Run the main script to activate the CrewAI system:

```
python agents/rasp_pi.py
```

This will initiate the AI agent crew to:
1. Provide hardware connection guidance
2. Generate Python code for the web application
3. Refactor the code for improved quality
4. Create comprehensive tests

## Project Structure

```
.
├── agents/
│   ├── .env                  # Environment variables (OpenAI API key)
│   ├── rasp_pi.py            # Main script with CrewAI configuration
│   └── ... (other agent files)
├── hardware/                 # Hardware connection scripts and documentation
├── web_app/                  # Python web application for data visualization
├── tests/                    # Test suite using pytest
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- CrewAI framework
- OpenAI for providing the AI models
- Raspberry Pi Foundation
