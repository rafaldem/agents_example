"""
CRAFT Sequential Prompt Generation System
Implementation of Context, Role, Action, Format, Target Audience framework
for structured prompt engineering with multi-agent coordination.
"""


import logging
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Define available agent types for the CRAFT system."""
    HARDWARE = "hardware"
    WEB_DEV = "web_dev"
    REFACTOR = "refactor"
    SW_TESTER = "sw_tester"
    DEVOPS = "devops"


class ExecutionStatus(Enum):
    """Execution status for tracking agent results."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CraftPromptComponent:
    """
    Individual component of the CRAFT framework.

    Attributes:
        context: Situational background and environmental factors
        role: Specific expertise and authority perspective
        action: Detailed step-by-step instructions
        format: Expected output structure and presentation
        target_audience: Intended recipients and their expertise level
    """
    context: str = ""
    role: str = ""
    action: str = ""
    format: str = ""
    target_audience: str = ""

    def validate(self) -> List[str]:
        """
        Validate that all required CRAFT components are present.

        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        if not self.context.strip():
            errors.append("Context component is required")
        if not self.role.strip():
            errors.append("Role component is required")
        if not self.action.strip():
            errors.append("Action component is required")
        if not self.format.strip():
            errors.append("Format component is required")
        if not self.target_audience.strip():
            errors.append("Target Audience component is required")
        return errors


@dataclass
class CraftPromptTemplate:
    """
    Template for generating structured CRAFT prompts.

    Attributes:
        agent_type: Type of agent this template is designed for
        craft_components: CRAFT framework components
        additional_context: Additional contextual information
        metadata: Template metadata and configuration
    """
    agent_type: AgentType
    craft_components: CraftPromptComponent
    additional_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def generate_prompt(self) -> str:
        """
        Generate a formatted prompt using CRAFT structure.

        Returns:
            Formatted prompt string following CRAFT methodology
        """
        validation_errors = self.craft_components.validate()
        if validation_errors:
            raise ValueError(f"Invalid CRAFT components: {', '.join(validation_errors)}")

        prompt_sections = [
            f"**CONTEXT:** {self.craft_components.context}",
            f"**ROLE:** {self.craft_components.role}",
            f"**ACTION:** {self.craft_components.action}",
            f"**FORMAT:** {self.craft_components.format}",
            f"**TARGET AUDIENCE:** {self.craft_components.target_audience}"
        ]

        # Add additional context if present
        if self.additional_context:
            context_items = []
            for key, value in self.additional_context.items():
                if value:
                    context_items.append(f"- {key}: {value}")

            if context_items:
                prompt_sections.insert(1, f"**ADDITIONAL CONTEXT:**\n" + "\n".join(context_items))

        return "\n\n".join(prompt_sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            'agent_type': self.agent_type.value,
            'craft_components': asdict(self.craft_components),
            'additional_context': self.additional_context,
            'metadata': self.metadata
        }


@dataclass
class ExecutionResult:
    """
    Result from executing a single CRAFT prompt.

    Attributes:
        agent_type: Type of agent that executed the prompt
        prompt: The generated CRAFT prompt
        response: LLM response to the prompt
        execution_time: Time taken to execute the prompt
        status: Execution status
        timestamp: When the execution occurred
        error_message: Error details if execution failed
        metadata: Additional execution metadata
    """
    agent_type: AgentType
    prompt: str
    response: str
    execution_time: float
    status: ExecutionStatus
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'agent_type': self.agent_type.value,
            'prompt': self.prompt,
            'response': self.response,
            'execution_time': self.execution_time,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class CraftConfiguration:
    """
    Configuration management for CRAFT prompt generation system.

    Manages environment variables and agent configurations following
    the CRAFT framework structure.
    """

    def __init__(self, env_vars: Dict[str, str] = None):
        """
        Initialize configuration from environment variables.

        Args:
            env_vars: Dictionary of environment variables (for testing)
        """
        import os
        self.env_vars = env_vars or dict(os.environ)
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Validate that required configuration parameters are present."""
        required_vars = [
            'TOPIC',
            'TOPIC_WEBPAGE_URL',
        ]

        missing_vars = [var for var in required_vars if not self.env_vars.get(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def get_agent_config(self, agent_type: AgentType) -> Dict[str, str]:
        """
        Get CRAFT configuration for a specific agent type.

        Args:
            agent_type: The agent type to get configuration for

        Returns:
            Dictionary containing CRAFT component configurations
        """
        prefix_map = {
            AgentType.HARDWARE: "HW_AGENT",
            AgentType.WEB_DEV: "WEB_DEV_AGENT",
            AgentType.REFACTOR: "REFACTOR_AGENT",
            AgentType.SW_TESTER: "SW_TESTER_AGENT",
            AgentType.DEVOPS: "DEVOPS_AGENT"
        }

        prefix = prefix_map.get(agent_type, agent_type.value.upper() + "_AGENT")

        return {
            'context': self.env_vars.get(f'{prefix}_CONTEXT', ''),
            'role': self.env_vars.get(f'{prefix}_ROLE', ''),
            'action': self.env_vars.get(f'{prefix}_ACTION', ''),
            'format': self.env_vars.get(f'{prefix}_FORMAT', ''),
            'target_audience': self.env_vars.get(f'{prefix}_TARGET_AUDIENCE', ''),
            'task_description': self.env_vars.get(f'{prefix}_TASK_DESCRIPTION', ''),
            'expected_output': self.env_vars.get(f'{prefix}_EXPECTED_OUTPUT', '')
        }

    @property
    def topic(self) -> str:
        """Get the main project topic."""
        return self.env_vars.get('TOPIC', '')

    @property
    def topic_url(self) -> str:
        """Get the topic reference URL."""
        return self.env_vars.get('TOPIC_WEBPAGE_URL', '')

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get LLM model configuration."""
        return {
            'model_name': self.env_vars.get('AGENT_MODEL_NAME', 'claude-3-sonnet-20240229'),
            'temperature': float(self.env_vars.get('AGENT_TEMPERATURE', '0.3')),
            'max_tokens': int(self.env_vars.get('AGENT_MAX_TOKENS', '8192')),
            'top_p': float(self.env_vars.get('AGENT_TOP_P', '1.0')),
            'max_retries': int(self.env_vars.get('AGENT_MAX_RETRIES', '5'))
        }


class SequentialCraftGenerator:
    """
    Generates CRAFT-structured prompts for sequential execution across multiple agents.

    This class creates specialized prompts for different agent types while maintaining
    the CRAFT framework consistency and enabling context passing between sequential
    agent executions.
    """

    def __init__(self, config: CraftConfiguration):
        """
        Initialize the generator with configuration.

        Args:
            config: CRAFT configuration instance
        """
        self.config = config

    def create_hardware_prompt(self, specific_objective: str = None,
                               previous_results: List[ExecutionResult] = None) -> CraftPromptTemplate:
        """
        Create a hardware-focused CRAFT prompt.

        Args:
            specific_objective: Specific objective to override default
            previous_results: Results from previous agent executions

        Returns:
            CRAFT prompt template for hardware agent
        """
        agent_config = self.config.get_agent_config(AgentType.HARDWARE)

        # Build context from configuration and previous results
        context_parts = [
            agent_config['context'],
            f"The system involves {self.config.topic}.",
            f"Reference documentation: {self.config.topic_url}"
        ]

        if previous_results:
            context_parts.append(f"Previous analysis from {len(previous_results)} preceding agents has been completed.")

        craft_components = CraftPromptComponent(
            context=" ".join(context_parts),
            role=agent_config['role'] or "You are a hardware engineer with 15 years of experience specializing in IoT devices, particularly Raspberry Pi configurations for sensor data collection systems.",
            action=agent_config['action'] or """
1. Analyze the specific requirements of connecting 1-wire temperature sensors to a Raspberry Pi Zero W
2. Explain the optimal hardware configuration including pin connections, power considerations, and limitations
3. Recommend appropriate libraries and drivers for interfacing with the sensors
4. Suggest best practices for ensuring reliable data collection over extended periods
5. Address considerations for multiple sensor connections and scaling requirements
6. Provide troubleshooting guidelines for common hardware issues
7. Specify environmental protection and housing requirements
""",
            format=agent_config['format'] or "Provide your response with clear sections for hardware setup, software requirements, implementation examples, and troubleshooting procedures. Include technical specifications, wiring diagrams references, and code snippets where appropriate.",
            target_audience=agent_config['target_audience'] or "Technical implementers including embedded systems engineers, IoT developers, and hardware enthusiasts with intermediate to advanced electronics knowledge who need practical, actionable guidance for reliable sensor deployment."
        )

        additional_context = {
            'specific_objective': specific_objective or "Design and configure Raspberry Pi Zero W hardware setup for reliable 1-wire temperature sensor data collection",
            'hardware_constraints': "Raspberry Pi Zero W resource limitations and power efficiency requirements",
            'sensor_specifications': "1-wire temperature sensors (DS18B20) with multiple sensor support",
            'reliability_requirements': "Continuous operation for weeks/months with minimal maintenance"
        }

        return CraftPromptTemplate(
            agent_type=AgentType.HARDWARE,
            craft_components=craft_components,
            additional_context=additional_context,
            metadata={'task_description': agent_config['task_description']}
        )

    def create_web_dev_prompt(self, previous_results: List[ExecutionResult] = None) -> CraftPromptTemplate:
        """Create a web development CRAFT prompt incorporating previous results."""
        agent_config = self.config.get_agent_config(AgentType.WEB_DEV)

        context_parts = [agent_config['context']]

        if previous_results:
            hw_results = [r for r in previous_results if r.agent_type == AgentType.HARDWARE]
            if hw_results:
                context_parts.append(f"Hardware configuration analysis has been completed. Key hardware considerations: {hw_results[-1].response[:300]}...")

        craft_components = CraftPromptComponent(
            context=" ".join(context_parts),
            role=agent_config['role'] or "You are a senior Python web developer with 10+ years of experience in data visualization frameworks, specializing in time-series data presentation and responsive dashboard design.",
            action=agent_config['action'] or """
1. Evaluate Python web frameworks most suitable for temperature data visualization with Taipy
2. Design responsive interface architecture for displaying temperature time-series data
3. Implement time period selection functionality (daily, weekly, monthly, yearly views)
4. Develop efficient data retrieval patterns that integrate with existing SQL database API
5. Optimize application performance for various devices and concurrent user access
6. Ensure code quality through proper structure, documentation, and maintainability practices
7. Create export functionality for data and visualizations
8. Implement real-time data refresh capabilities
""",
            format=agent_config['format'] or "Structure your solution with distinct sections for architecture overview, component design, implementation examples, and deployment considerations. Include code snippets, configuration examples, and performance optimization strategies.",
            target_audience=agent_config['target_audience'] or "Python developers, web application architects, and data visualization specialists who need to implement production-ready temperature monitoring dashboards with professional user experience standards."
        )

        additional_context = {
            'framework_focus': "Taipy-based web application development",
            'data_source': "SQL database with existing API backend",
            'visualization_requirements': "Time-series charts with multiple time period views",
            'performance_targets': "Responsive interface supporting multiple concurrent users",
            'integration_requirements': "RESTful API integration for data retrieval"
        }

        return CraftPromptTemplate(
            agent_type=AgentType.WEB_DEV,
            craft_components=craft_components,
            additional_context=additional_context,
            metadata={'task_description': agent_config['task_description']}
        )

    def create_refactor_prompt(self, content_to_refactor: str, content_type: str,
                               previous_results: List[ExecutionResult] = None) -> CraftPromptTemplate:
        """Create a refactoring CRAFT prompt for improving generated content."""
        agent_config = self.config.get_agent_config(AgentType.REFACTOR)

        context_parts = [
            agent_config['context'],
            f"The following {content_type} content requires refactoring and enhancement:",
            f"```\n{content_to_refactor[:2500]}{'...' if len(content_to_refactor) > 2500 else ''}\n```"
        ]

        craft_components = CraftPromptComponent(
            context=" ".join(context_parts),
            role=agent_config['role'] or "You are a software architect with 12 years specializing in Python code quality, performance optimization, and maintainable system design with expertise in refactoring enterprise applications.",
            action=agent_config['action'] or f"""
1. Analyze the current {content_type} structure, identifying improvement opportunities and anti-patterns
2. Evaluate technical accuracy, implementation feasibility, and best practice adherence
3. Assess clarity, organization, and logical flow of the content
4. Identify gaps in error handling, edge cases, and robustness considerations
5. Suggest specific refactoring steps with clear rationales and implementation guidance
6. Provide enhanced examples and improved technical specifications
7. Ensure professional presentation and comprehensive documentation quality
8. Maintain original intent while significantly improving practical implementation value
""",
            format=agent_config['format'] or f"Present your enhanced {content_type} with distinct sections for overall assessment, specific improvements, refactored implementation, and implementation rationale. Include before/after comparisons where applicable and reference relevant design patterns or best practices.",
            target_audience=agent_config['target_audience'] or "Software engineers, system architects, and technical leads who need production-ready, maintainable solutions with clear implementation guidance and professional documentation standards."
        )

        additional_context = {
            'content_type': content_type,
            'refactoring_focus': "Quality, clarity, and implementation effectiveness enhancement",
            'technical_standards': "Industry best practices and proven methodologies",
            'improvement_scope': "Incremental improvements that enhance practical implementation value"
        }

        return CraftPromptTemplate(
            agent_type=AgentType.REFACTOR,
            craft_components=craft_components,
            additional_context=additional_context,
            metadata={'content_type': content_type, 'original_length': len(content_to_refactor)}
        )

    def create_testing_prompt(self, previous_results: List[ExecutionResult] = None) -> CraftPromptTemplate:
        """Create a testing-focused CRAFT prompt."""
        agent_config = self.config.get_agent_config(AgentType.SW_TESTER)

        context_parts = [agent_config['context']]

        if previous_results:
            web_results = [r for r in previous_results if r.agent_type == AgentType.WEB_DEV]
            if web_results:
                context_parts.append(f"Web development specifications from previous analysis provide implementation context for comprehensive testing design.")

        craft_components = CraftPromptComponent(
            context=" ".join(context_parts),
            role=agent_config['role'] or "You are a quality assurance engineer with 8 years of experience specializing in test-driven development for Python applications, with expertise in pytest frameworks and critical system testing.",
            action=agent_config['action'] or """
1. Analyze system requirements for comprehensive test case design covering normal and edge cases
2. Design pytest-based unit tests with appropriate fixtures, mocking, and parameterization
3. Implement integration tests for component interaction validation
4. Create functional tests using Selenium for web interface validation and user workflows
5. Develop test data management strategies and mock implementations for external dependencies
6. Establish performance benchmarks and assertions for data processing components
7. Design test automation suitable for CI/CD pipeline integration
8. Create comprehensive test documentation and coverage analysis procedures
""",
            format=agent_config['format'] or "Structure your testing solution with sections for test strategy overview, test implementation details, fixture design, mock strategies, and CI/CD integration guidance. Include complete test examples and coverage analysis approaches.",
            target_audience=agent_config['target_audience'] or "Quality assurance engineers, software developers implementing TDD practices, and DevOps engineers establishing automated testing pipelines for production deployment validation."
        )

        additional_context = {
            'testing_framework': "pytest with comprehensive fixture design",
            'test_types': "Unit tests, integration tests, functional tests with Selenium",
            'coverage_target': "Minimum 90% code coverage for critical components",
            'automation_focus': "CI/CD pipeline integration and automated validation"
        }

        return CraftPromptTemplate(
            agent_type=AgentType.SW_TESTER,
            craft_components=craft_components,
            additional_context=additional_context,
            metadata={'task_description': agent_config['task_description']}
        )

    def create_devops_prompt(self, previous_results: List[ExecutionResult] = None) -> CraftPromptTemplate:
        """Create a DevOps-focused CRAFT prompt incorporating previous development results."""
        agent_config = self.config.get_agent_config(AgentType.DEVOPS)

        context_parts = [agent_config['context']]

        if previous_results:
            relevant_results = [r for r in previous_results if r.agent_type in [AgentType.WEB_DEV, AgentType.HARDWARE, AgentType.SW_TESTER]]
            if relevant_results:
                context_parts.append(f"Development context from {len(relevant_results)} preceding agents provides comprehensive system understanding for infrastructure design.")

        craft_components = CraftPromptComponent(
            context=" ".join(context_parts),
            role=agent_config['role'] or "You are a DevOps architect with 14 years of experience in containerization and infrastructure automation, specializing in IoT system deployments and edge device optimization.",
            action=agent_config['action'] or """
1. Analyze requirements for containerizing data collection and web visualization components
2. Design Docker-based infrastructure optimized for Raspberry Pi ARM architecture deployments
3. Develop multi-stage Dockerfiles for efficient image size and resource utilization
4. Create container orchestration strategy using Docker Compose for service coordination
5. Implement automated CI/CD pipelines with build, testing, and deployment automation
6. Establish comprehensive monitoring, logging, and health check mechanisms
7. Design resource optimization strategies for limited Pi hardware constraints
8. Create deployment, maintenance, and recovery documentation and procedures
""",
            format=agent_config['format'] or "Provide your infrastructure solution with clear sections for architecture overview, containerization strategy, deployment workflow, monitoring setup, and operational procedures. Include configuration files, deployment scripts, and troubleshooting guides.",
            target_audience=agent_config['target_audience'] or "DevOps engineers, system administrators, and infrastructure architects implementing edge computing solutions with emphasis on resource-constrained environments and automated deployment processes."
        )

        additional_context = {
            'target_platform': "Raspberry Pi (ARM architecture) with resource optimization",
            'containerization': "Docker with multi-stage builds for efficiency",
            'orchestration': "Docker Compose for local deployment coordination",
            'monitoring_focus': "Comprehensive logging, health checks, and performance monitoring"
        }

        return CraftPromptTemplate(
            agent_type=AgentType.DEVOPS,
            craft_components=craft_components,
            additional_context=additional_context,
            metadata={'task_description': agent_config['task_description']}
        )


class CraftLLMRunner:
    """
    Enhanced LLM runner for sequential CRAFT prompt execution with comprehensive result management.

    Coordinates the execution of CRAFT-structured prompts across multiple agents,
    maintaining execution context and providing detailed result tracking.
    """

    def __init__(self, config: CraftConfiguration):
        """
        Initialize the LLM runner with configuration.

        Args:
            config: CRAFT configuration instance
        """
        self.config = config
        self.craft_generator = SequentialCraftGenerator(config)
        self.execution_history: List[ExecutionResult] = []

        # Initialize LLM client based on configuration
        # Note: In production, this would initialize the actual LLM client
        # self.client = self._initialize_llm_client()

    def _initialize_llm_client(self):
        """Initialize the LLM client based on configuration."""
        # This would initialize the actual LLM client in production
        # For example: Together(), OpenAI(), etc.
        logger.info("LLM client initialization (placeholder)")
        return None

    async def execute_prompt(self, prompt_template: CraftPromptTemplate) -> ExecutionResult:
        """
        Execute a single CRAFT prompt and return structured result.

        Args:
            prompt_template: CRAFT prompt template to execute

        Returns:
            Execution result with response and metadata
        """
        start_time = time.time()

        try:
            prompt = prompt_template.generate_prompt()
            logger.info(f"Executing {prompt_template.agent_type.value} CRAFT prompt")

            # In production, this would call the actual LLM
            # response = await self.client.chat.completions.create(...)

            # Placeholder response for demonstration
            response = f"[CRAFT Response for {prompt_template.agent_type.value}]\n{prompt[:200]}..."

            execution_time = time.time() - start_time

            result = ExecutionResult(
                agent_type=prompt_template.agent_type,
                prompt=prompt,
                response=response,
                execution_time=execution_time,
                status=ExecutionStatus.SUCCESS,
                metadata={
                    'prompt_template_metadata': prompt_template.metadata,
                    'response_length': len(response),
                    'model_config': self.config.model_config
                }
            )

            self.execution_history.append(result)
            logger.info(f"Successfully executed {prompt_template.agent_type.value} prompt in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time

            error_result = ExecutionResult(
                agent_type=prompt_template.agent_type,
                prompt=prompt_template.generate_prompt() if prompt_template else "",
                response="",
                execution_time=execution_time,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                metadata={'model_config': self.config.model_config}
            )

            logger.error(f"Error executing {prompt_template.agent_type.value}: {e}")
            self.execution_history.append(error_result)
            return error_result

    async def execute_development_pipeline(self, enable_refactoring: bool = True) -> List[ExecutionResult]:
        """
        Execute the complete development pipeline with CRAFT methodology.

        Args:
            enable_refactoring: Whether to include refactoring phases

        Returns:
            List of execution results from all pipeline phases
        """
        pipeline_results = []

        # Phase 1: Hardware Analysis
        logger.info("Phase 1: Starting hardware analysis with CRAFT framework")
        hw_prompt = self.craft_generator.create_hardware_prompt()
        hw_result = await self.execute_prompt(hw_prompt)
        pipeline_results.append(hw_result)

        # Phase 2: Web Development
        logger.info("Phase 2: Starting web development with CRAFT framework")
        web_prompt = self.craft_generator.create_web_dev_prompt(pipeline_results)
        web_result = await self.execute_prompt(web_prompt)
        pipeline_results.append(web_result)

        # Phase 3: Testing Strategy
        logger.info("Phase 3: Starting testing strategy with CRAFT framework")
        test_prompt = self.craft_generator.create_testing_prompt(pipeline_results)
        test_result = await self.execute_prompt(test_prompt)
        pipeline_results.append(test_result)

        # Phase 4: DevOps Infrastructure
        logger.info("Phase 4: Starting DevOps infrastructure with CRAFT framework")
        devops_prompt = self.craft_generator.create_devops_prompt(pipeline_results)
        devops_result = await self.execute_prompt(devops_prompt)
        pipeline_results.append(devops_result)

        # Refactoring Phases (if enabled)
        if enable_refactoring:
            refactoring_targets = [
                (web_result, "Web Development Implementation"),
                (test_result, "Testing Strategy"),
                (devops_result, "DevOps Infrastructure")
            ]

            for i, (target_result, content_type) in enumerate(refactoring_targets, 1):
                if target_result.status == ExecutionStatus.SUCCESS:
                    logger.info(f"Refactoring Phase {i}: Enhancing {content_type}")
                    refactor_prompt = self.craft_generator.create_refactor_prompt(
                        target_result.response, content_type, pipeline_results
                    )
                    refactor_result = await self.execute_prompt(refactor_prompt)
                    pipeline_results.append(refactor_result)

        logger.info(f"Pipeline execution completed. {len(pipeline_results)} total operations performed.")
        return pipeline_results

    def generate_execution_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive execution report with detailed analytics.

        Returns:
            Dictionary containing execution metrics and analysis
        """
        if not self.execution_history:
            return {"message": "No executions completed.", "results": []}

        successful_executions = [r for r in self.execution_history if r.status == ExecutionStatus.SUCCESS]
        failed_executions = [r for r in self.execution_history if r.status == ExecutionStatus.FAILED]

        # Calculate execution metrics
        total_time = sum(r.execution_time for r in self.execution_history)
        avg_execution_time = total_time / len(self.execution_history) if self.execution_history else 0

        # Group by agent type
        agent_performance = {}
        for agent_type in AgentType:
            agent_results = [r for r in self.execution_history if r.agent_type == agent_type]
            if agent_results:
                agent_performance[agent_type.value] = {
                    'total_executions': len(agent_results),
                    'successful': len([r for r in agent_results if r.status == ExecutionStatus.SUCCESS]),
                    'failed': len([r for r in agent_results if r.status == ExecutionStatus.FAILED]),
                    'avg_execution_time': sum(r.execution_time for r in agent_results) / len(agent_results),
                    'total_response_length': sum(len(r.response) for r in agent_results)
                }

        return {
            "execution_summary": {
                "total_executions": len(self.execution_history),
                "successful_executions": len(successful_executions),
                "failed_executions": len(failed_executions),
                "success_rate": len(successful_executions) / len(self.execution_history) * 100,
                "total_execution_time": total_time,
                "average_execution_time": avg_execution_time
            },
            "agent_performance": agent_performance,
            "execution_timeline": [result.to_dict() for result in self.execution_history],
            "configuration_used": {
                "topic": self.config.topic,
                "model_config": self.config.model_config
            }
        }

    def save_results(self, output_path: str = "craft_execution_results.json") -> None:
        """
        Save execution results to file.

        Args:
            output_path: Path to save results file
        """
        report = self.generate_execution_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Execution results saved to {output_path}")


# Example usage and main execution
if __name__ == "__main__":
    async def main():
        """Example implementation of the CRAFT Sequential Prompt Generation System."""

        # Example environment configuration
        example_env = {
            'TOPIC': 'Web application based on Taipy package displaying daily, weekly, monthly or yearly historical data collected using Raspberry Pi Zero W 1-wire temperature sensor which are stored in a database.',
            'TOPIC_WEBPAGE_URL': 'https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#raspberry-pi-zero-w',
            'HW_AGENT_CONTEXT': 'You are tasked with providing expertise on Raspberry Pi Zero W hardware configuration for a temperature monitoring system using 1-wire sensors.',
            'HW_AGENT_ROLE': 'You are a hardware engineer with 15 years of experience specializing in IoT devices.',
            'WEB_DEV_AGENT_CONTEXT': 'You are tasked with developing a Python-based web frontend application that visualizes temperature data.',
            'WEB_DEV_AGENT_ROLE': 'You are a senior Python web developer with 10+ years of experience in data visualization frameworks.',
            'AGENT_MODEL_NAME': 'claude-3-sonnet-20240229',
            'AGENT_TEMPERATURE': '0.3',
            'AGENT_MAX_TOKENS': '8192'
        }

        try:
            # Initialize configuration
            config = CraftConfiguration(example_env)

            # Initialize CRAFT runner
            runner = CraftLLMRunner(config)

            # Execute the complete pipeline
            logger.info("Starting CRAFT Sequential Prompt Generation System execution")
            results = await runner.execute_development_pipeline(enable_refactoring=True)

            # Generate and save comprehensive report
            report = runner.generate_execution_report()
            runner.save_results("craft_execution_results.json")

            # Print summary
            print("\n" + "="*60)
            print("CRAFT SEQUENTIAL EXECUTION SUMMARY")
            print("="*60)
            print(f"Total Executions: {report['execution_summary']['total_executions']}")
            print(f"Success Rate: {report['execution_summary']['success_rate']:.1f}%")
            print(f"Total Time: {report['execution_summary']['total_execution_time']:.2f}s")
            print(f"Average Time per Execution: {report['execution_summary']['average_execution_time']:.2f}s")
            print("="*60)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    # Run the example
    asyncio.run(main())