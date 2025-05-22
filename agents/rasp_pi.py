"""
Sequential Prompt Generation System using COMPASS-DEV Framework
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json

from setup import *
from together import AsyncTogether, Together

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Define available agent types for the system."""
    HARDWARE = "hardware"
    WEB_DEV = "web_dev"
    REFACTOR = "refactor"
    SW_TESTER = "sw_tester"
    DEVOPS = "devops"


@dataclass
class CompassPromptComponent:
    """Individual component of the COMPASS-DEV framework."""
    context: str = ""
    objective: str = ""
    method: str = ""
    parameters: str = ""
    assessment: str = ""
    specifications: str = ""
    scope: str = ""


@dataclass
class PromptTemplate:
    """Template for generating structured prompts."""
    agent_type: AgentType
    compass_components: CompassPromptComponent
    additional_context: Dict[str, Any] = field(default_factory=dict)

    def generate_prompt(self) -> str:
        """Generate a formatted prompt using COMPASS-DEV structure."""
        prompt_parts = []

        if self.compass_components.context:
            prompt_parts.append(f"CONTEXT: {self.compass_components.context}")

        if self.compass_components.objective:
            prompt_parts.append(f"OBJECTIVE: {self.compass_components.objective}")

        if self.compass_components.method:
            prompt_parts.append(f"METHOD: {self.compass_components.method}")

        if self.compass_components.parameters:
            prompt_parts.append(f"PARAMETERS: {self.compass_components.parameters}")

        if self.compass_components.assessment:
            prompt_parts.append(f"ASSESSMENT: {self.compass_components.assessment}")

        if self.compass_components.specifications:
            prompt_parts.append(f"SPECIFICATIONS: {self.compass_components.specifications}")

        if self.compass_components.scope:
            prompt_parts.append(f"SCOPE: {self.compass_components.scope}")

        return "\n\n".join(prompt_parts)


@dataclass
class ExecutionResult:
    """Result from executing a single prompt."""
    agent_type: AgentType
    prompt: str
    response: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class SequentialPromptGenerator:
    """Generates prompts using COMPASS-DEV framework for sequential execution."""

    def __init__(self):
        self.agent_configs = self._load_agent_configurations()

    def _load_agent_configurations(self) -> Dict[AgentType, Dict[str, str]]:
        """Load agent configurations from environment variables."""
        return {
            AgentType.HARDWARE: {
                "context": ENVS.get('HW_AGENT_CONTEXT', ''),
                "role": ENVS.get('HW_AGENT_ROLE', ''),
                "action": ENVS.get('HW_AGENT_ACTION', ''),
                "format": ENVS.get('HW_AGENT_FORMAT', ''),
                "task_description": ENVS.get('HW_EXPERT_TASK_DESCRIPTION', ''),
                "expected_output": ENVS.get('HW_EXPERT_TASK_EXPECTED_OUTPUT', '')
            },
            AgentType.WEB_DEV: {
                "context": ENVS.get('WEB_DEV_CONTEXT', ''),
                "role": ENVS.get('WEB_DEV_ROLE', ''),
                "action": ENVS.get('WEB_DEV_ACTION', ''),
                "format": ENVS.get('WEB_DEV_FORMAT', ''),
                "task_description": ENVS.get('WEB_DEVELOPER_TASK_DESCRIPTION', ''),
                "expected_output": ENVS.get('WEB_DEVELOPER_TASK_EXPECTED_OUTPUT', '')
            },
            AgentType.REFACTOR: {
                "context": ENVS.get('REFACTOR_CONTEXT', ''),
                "role": ENVS.get('REFACTOR_EXPERT_ROLE', ''),
                "action": ENVS.get('REFACTOR_EXPERT_ACTION', ''),
                "format": ENVS.get('REFACTOR_EXPERT_FORMAT', ''),
                "task_description": ENVS.get('REFACTOR_TASK_DESCRIPTION', ''),
                "expected_output": ENVS.get('REFACTOR_TASK_EXPECTED_OUTPUT', '')
            },
            AgentType.SW_TESTER: {
                "context": ENVS.get('SW_TESTER_CONTEXT', ''),
                "role": ENVS.get('SW_TESTER_ROLE', ''),
                "action": ENVS.get('SW_TESTER_ACTION', ''),
                "format": ENVS.get('SW_TESTER_FORMAT', ''),
                "task_description": ENVS.get('SW_TESTER_TASK_DESCRIPTION', ''),
                "expected_output": ENVS.get('SW_TESTER_TASK_EXPECTED_OUTPUT', '')
            },
            AgentType.DEVOPS: {
                "context": ENVS.get('DEVOPS_CONTEXT', ''),
                "role": ENVS.get('DEVOPS_ROLE', ''),
                "action": ENVS.get('DEVOPS_ACTION', ''),
                "format": ENVS.get('DEVOPS_FORMAT', ''),
                "task_description": ENVS.get('DEVOPS_TASK_DESCRIPTION', ''),
                "expected_output": ENVS.get('DEVOPS_TASK_EXPECTED_OUTPUT', '')
            }
        }

    def create_hardware_prompt(self, specific_objective: str = None) -> PromptTemplate:
        """Create a hardware-focused prompt using COMPASS-DEV framework."""
        config = self.agent_configs[AgentType.HARDWARE]

        compass_components = CompassPromptComponent(
            context=f"{config['context']} The system involves {TOPIC}. Reference documentation: {TOPIC_WEBPAGE_URL}",
            objective=specific_objective or "Design and configure Raspberry Pi Zero W hardware setup for reliable 1-wire temperature sensor data collection",
            method=config['action'],
            parameters="""
            - Target hardware: Raspberry Pi Zero W
            - Sensor type: 1-wire temperature sensors (DS18B20)
            - Power requirements: Efficient power consumption for continuous operation
            - Data collection frequency: Configurable intervals
            - Environmental considerations: Indoor deployment
            - Scalability: Support for multiple sensors
            """,
            assessment="""
            The solution will be evaluated based on:
            - Reliable sensor readings with minimal errors
            - Stable operation over extended periods (weeks/months)
            - Efficient resource utilization on Raspberry Pi Zero W
            - Clear documentation and setup instructions
            - Compatibility with standard Python libraries
            """,
            specifications="""
            - GPIO pin assignments for 1-wire communication
            - Pull-up resistor configuration (4.7kΩ)
            - Power supply specifications and connections
            - Housing and environmental protection requirements
            - Library dependencies and driver installation
            - Error handling for sensor disconnections
            """,
            scope="""
            - INCLUDE: Hardware wiring, driver configuration, basic testing procedures
            - EXCLUDE: Database design, web interface development, advanced analytics
            - Focus on hardware reliability and data acquisition accuracy
            """
        )

        return PromptTemplate(AgentType.HARDWARE, compass_components)

    def create_web_dev_prompt(self, previous_results: List[ExecutionResult] = None) -> PromptTemplate:
        """Create a web development prompt incorporating previous results."""
        config = self.agent_configs[AgentType.WEB_DEV]

        context_addition = ""
        if previous_results:
            hw_results = [r for r in previous_results if r.agent_type == AgentType.HARDWARE]
            if hw_results:
                context_addition = f" Hardware configuration from previous analysis: {hw_results[-1].response[:200]}..."

        compass_components = CompassPromptComponent(
            context=f"{config['context']}{context_addition}",
            objective="Develop a Taipy-based web application for visualizing temperature sensor data with multiple time period views",
            method=config['action'],
            parameters="""
            - Framework: Taipy for Python web applications
            - Database: SQL database with existing API backend
            - Time periods: Daily, weekly, monthly, yearly views
            - Performance: Responsive interface supporting multiple concurrent users
            - Compatibility: Modern web browsers, mobile-responsive design
            - Data refresh: Real-time or near-real-time updates
            """,
            assessment="""
            The web application will be evaluated on:
            - Intuitive user interface for time period selection
            - Efficient data loading and visualization performance
            - Proper integration with existing API backend
            - Visual appeal and professional appearance
            - Cross-browser compatibility and responsiveness
            - Code quality and maintainability
            """,
            specifications="""
            - Time-series visualization components using Taipy
            - RESTful API integration for data retrieval
            - Interactive charts with zoom and pan capabilities
            - Date range selection interface
            - Export functionality for data and visualizations
            - Error handling for API connectivity issues
            - Caching strategy for improved performance
            """,
            scope="""
            - INCLUDE: Web interface, data visualization, API integration, responsive design
            - EXCLUDE: Database schema design, hardware configuration, DevOps deployment
            - Focus on user experience and data presentation effectiveness
            """
        )

        return PromptTemplate(AgentType.WEB_DEV, compass_components)

    def create_refactor_prompt(self, content_to_refactor: str, content_type: str, previous_results: List[ExecutionResult] = None) -> PromptTemplate:
        """Create a refactoring prompt for improving generated content."""
        config = self.agent_configs[AgentType.REFACTOR]

        compass_components = CompassPromptComponent(
            context=f"{config['context']} {content_type} content to refactor and improve:\n```\n{content_to_refactor[:2000]}...\n```",
            objective=f"Refactor and enhance the {content_type} content to improve quality, clarity, and implementation effectiveness",
            method=config['action'],
            parameters=f"""
            - Content type: {content_type} deliverable
            - Quality standards: Professional documentation and implementation clarity
            - Technical accuracy: Verify technical specifications and recommendations
            - Implementation feasibility: Ensure practical applicability
            - Documentation quality: Improve explanations and instructions
            - Best practices: Apply industry standards and proven methodologies
            """,
            assessment=f"""
            Refactored {content_type} content will be evaluated on:
            - Enhanced clarity and technical accuracy
            - Improved implementation guidance and specificity
            - Better organization and logical flow
            - Adherence to industry best practices
            - Practical applicability and completeness
            - Professional presentation and documentation quality
            """,
            specifications=f"""
            - Maintain technical accuracy while improving clarity
            - Enhance implementation details and step-by-step guidance
            - Improve code examples and configuration specifications
            - Add missing technical considerations and edge cases
            - Reorganize content for better logical flow
            - Provide comprehensive error handling and troubleshooting guidance
            """,
            scope=f"""
            - INCLUDE: Content improvement, technical enhancement, documentation clarity
            - EXCLUDE: Complete architectural redesign, fundamental approach changes
            - Focus on incremental improvements that enhance practical implementation value
            """
        )

        return PromptTemplate(AgentType.REFACTOR, compass_components)

    def create_testing_prompt(self, previous_results: List[ExecutionResult] = None) -> PromptTemplate:
        """Create a testing-focused prompt using COMPASS-DEV framework."""
        config = self.agent_configs[AgentType.SW_TESTER]

        context_addition = ""
        if previous_results:
            web_results = [r for r in previous_results if r.agent_type == AgentType.WEB_DEV]
            if web_results:
                context_addition = f" Web development specifications from previous analysis: {web_results[-1].response[:200]}..."

        compass_components = CompassPromptComponent(
            context=f"{config['context']}{context_addition}",
            objective="Design and implement comprehensive testing strategy for temperature monitoring system components",
            method=config['action'],
            parameters="""
            - Testing framework: pytest with appropriate fixtures for Python components
            - Test types: Unit tests, integration tests, functional tests with Selenium for web interface
            - Coverage target: Minimum 90% code coverage for critical components
            - Test data: Mock objects and test fixtures for reproducible tests
            - Continuous integration: Tests designed for automated CI/CD pipeline execution
            - Performance testing: Basic performance benchmarks for data processing and web interface
            """,
            assessment="""
            Test suite quality will be measured by:
            - Comprehensive coverage of normal and edge cases across all system components
            - Clear test naming conventions and comprehensive documentation
            - Efficient test execution time suitable for development workflow
            - Reliable test results across different deployment environments
            - Proper use of mocking for external dependencies and hardware interfaces
            - Integration with existing testing infrastructure and CI/CD processes
            """,
            specifications="""
            - Test file organization following pytest conventions and project structure
            - Parametrized tests for multiple input scenarios and configuration variations
            - Fixture design for test data setup, teardown, and hardware simulation
            - Mock implementations for external services, databases, and hardware interfaces
            - Selenium tests for web interface functionality and user interaction workflows
            - Performance assertions and benchmarks for data processing components
            """,
            scope="""
            - INCLUDE: Unit tests, integration tests, web interface tests, test fixtures, comprehensive documentation
            - EXCLUDE: Load testing, security penetration testing, manual testing procedures
            - Focus on automated testing that supports development workflow and deployment validation
            """
        )

        return PromptTemplate(AgentType.SW_TESTER, compass_components)

    def create_devops_prompt(self, previous_results: List[ExecutionResult] = None) -> PromptTemplate:
        """Create a DevOps-focused prompt incorporating previous development results."""
        config = self.agent_configs[AgentType.DEVOPS]

        context_addition = ""
        if previous_results:
            relevant_results = [r for r in previous_results if r.agent_type in [AgentType.WEB_DEV, AgentType.HARDWARE, AgentType.SW_TESTER]]
            if relevant_results:
                context_addition = f" Previous development context: {len(relevant_results)} components analyzed including hardware, web development, and testing specifications."

        compass_components = CompassPromptComponent(
            context=f"{config['context']}{context_addition}",
            objective="Design containerized deployment infrastructure for Raspberry Pi temperature monitoring system with comprehensive automation",
            method=config['action'],
            parameters="""
            - Target platform: Raspberry Pi (ARM architecture) with resource optimization
            - Containerization: Docker with multi-stage builds for efficient image size
            - Orchestration: Docker Compose for local deployment and service coordination
            - CI/CD: Automated build, testing, and deployment pipeline with rollback capabilities
            - Monitoring: Comprehensive logging, health checks, and performance monitoring
            - Resource constraints: Optimize for limited Pi resources while maintaining functionality
            """,
            assessment="""
            Infrastructure will be evaluated on:
            - Successful deployment and operation on Raspberry Pi hardware
            - Automated build, test, and deployment processes with error handling
            - Resource efficiency and system performance under operational load
            - Monitoring and logging effectiveness for troubleshooting and maintenance
            - Ease of maintenance, updates, and system recovery procedures
            - Documentation quality for deployment, operation, and troubleshooting procedures
            """,
            specifications="""
            - Multi-architecture Docker images supporting ARM64 and ARM7 architectures
            - Docker Compose configuration for complete application stack coordination
            - Environment variable management for flexible configuration across environments
            - Volume mounts for persistent data storage and configuration management
            - Network configuration for sensor communication and web interface access
            - Health checks, restart policies, and automated recovery mechanisms
            """,
            scope="""
            - INCLUDE: Containerization, deployment automation, monitoring setup, documentation
            - EXCLUDE: Cloud deployment platforms, advanced orchestration systems like Kubernetes
            - Focus on edge deployment optimized for Raspberry Pi environments and constraints
            """
        )

        return PromptTemplate(AgentType.DEVOPS, compass_components)


class SequentialLLMRunner:
    """Enhanced LLM runner for sequential prompt execution with refactoring capabilities."""

    def __init__(self, model: str = None):
        self.model = model or AGENT_MODEL_NAME
        self.client = Together()
        self.async_client = AsyncTogether()
        self.prompt_generator = SequentialPromptGenerator()
        self.execution_history: List[ExecutionResult] = []

    async def execute_prompt(self, prompt_template: PromptTemplate) -> ExecutionResult:
        """Execute a single prompt and return structured result."""
        import time
        start_time = time.time()

        try:
            prompt = prompt_template.generate_prompt()
            logger.info(f"Executing {prompt_template.agent_type.value} prompt")

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(AGENT_TEMPERATURE or 0.3),
                max_tokens=int(AGENT_MAX_TOKENS or 8192),
            )

            execution_time = time.time() - start_time
            result = ExecutionResult(
                agent_type=prompt_template.agent_type,
                prompt=prompt,
                response=response.choices[0].message.content,
                execution_time=execution_time,
                success=True
            )

            self.execution_history.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_result = ExecutionResult(
                agent_type=prompt_template.agent_type,
                prompt=prompt_template.generate_prompt(),
                response="",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

            logger.error(f"Error executing {prompt_template.agent_type.value}: {e}")
            self.execution_history.append(error_result)
            return error_result

    async def execute_development_pipeline(self) -> List[ExecutionResult]:
        """Execute the complete development pipeline with refactoring enhancements."""
        pipeline_results = []

        # Step 1: Hardware Analysis
        logger.info("Starting hardware analysis phase")
        hw_prompt = self.prompt_generator.create_hardware_prompt()
        hw_result = await self.execute_prompt(hw_prompt)
        pipeline_results.append(hw_result)

        # Step 2: Web Development (incorporating hardware results)
        logger.info("Starting web development phase")
        web_prompt = self.prompt_generator.create_web_dev_prompt(pipeline_results)
        web_result = await self.execute_prompt(web_prompt)
        pipeline_results.append(web_result)

        # Step 3: Testing Strategy
        logger.info("Starting testing strategy phase")
        test_prompt = self.prompt_generator.create_testing_prompt(pipeline_results)
        test_result = await self.execute_prompt(test_prompt)
        pipeline_results.append(test_result)

        # Step 4: DevOps Infrastructure
        logger.info("Starting DevOps infrastructure phase")
        devops_prompt = self.prompt_generator.create_devops_prompt(pipeline_results)
        devops_result = await self.execute_prompt(devops_prompt)
        pipeline_results.append(devops_result)

        # Step 5: Refactor Web Development Output
        if web_result.success:
            logger.info("Refactoring web development output")
            web_refactor_prompt = self.prompt_generator.create_refactor_prompt(
                web_result.response, "Web Development", pipeline_results
            )
            web_refactor_result = await self.execute_prompt(web_refactor_prompt)
            pipeline_results.append(web_refactor_result)

        # Step 6: Refactor Testing Strategy Output
        if test_result.success:
            logger.info("Refactoring testing strategy output")
            test_refactor_prompt = self.prompt_generator.create_refactor_prompt(
                test_result.response, "Testing Strategy", pipeline_results
            )
            test_refactor_result = await self.execute_prompt(test_refactor_prompt)
            pipeline_results.append(test_refactor_result)

        # Step 7: Refactor DevOps Infrastructure Output
        if devops_result.success:
            logger.info("Refactoring DevOps infrastructure output")
            devops_refactor_prompt = self.prompt_generator.create_refactor_prompt(
                devops_result.response, "DevOps Infrastructure", pipeline_results
            )
            devops_refactor_result = await self.execute_prompt(devops_refactor_prompt)
            pipeline_results.append(devops_refactor_result)

        return pipeline_results

    def generate_execution_report(self) -> str:
        """Generate a comprehensive report of all executions."""
        if not self.execution_history:
            return "No executions completed."

        report_parts = [
            "# Sequential Prompt Execution Report",
            f"Total executions: {len(self.execution_history)}",
            f"Successful executions: {sum(1 for r in self.execution_history if r.success)}",
            f"Failed executions: {sum(1 for r in self.execution_history if not r.success)}",
            f"Total execution time: {sum(r.execution_time for r in self.execution_history):.2f} seconds",
            ""
        ]

        # Group results by phase
        phase_groups = {
            "Initial Development": [AgentType.HARDWARE, AgentType.WEB_DEV, AgentType.SW_TESTER, AgentType.DEVOPS],
            "Refactoring Phase": []  # Will contain refactor results
        }

        initial_results = [r for r in self.execution_history if r.agent_type in phase_groups["Initial Development"]]
        refactor_results = [r for r in self.execution_history if r.agent_type == AgentType.REFACTOR]

        report_parts.append("## Initial Development Phase")
        for result in initial_results:
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            report_parts.extend([
                f"### {result.agent_type.value.title()} {status}",
                f"Execution time: {result.execution_time:.2f} seconds",
                f"Response length: {len(result.response)} characters",
                ""
            ])

            if not result.success and result.error_message:
                report_parts.append(f"Error: {result.error_message}")
                report_parts.append("")

        if refactor_results:
            report_parts.append("## Refactoring Enhancement Phase")
            for i, result in enumerate(refactor_results, 1):
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                report_parts.extend([
                    f"### Refactoring Pass {i} {status}",
                    f"Execution time: {result.execution_time:.2f} seconds",
                    f"Response length: {len(result.response)} characters",
                    ""
                ])

                if not result.success and result.error_message:
                    report_parts.append(f"Error: {result.error_message}")
                    report_parts.append("")

        return "\n".join(report_parts)


if __name__ == "__main__":
    async def main():
        # Execute the enhanced development pipeline
        runner = SequentialLLMRunner()

        # Execute the complete pipeline with refactoring
        results = await runner.execute_development_pipeline()

        # Generate and print comprehensive report
        report = runner.generate_execution_report()
        print(report)

        # Save results to file with enhanced metadata
        with open('sequential_execution_results.json', 'w') as f:
            json.dump([{
                'agent_type': r.agent_type.value,
                'success': r.success,
                'execution_time': r.execution_time,
                'response_length': len(r.response),
                'error_message': r.error_message,
                'timestamp': None  # Could add timestamp if needed
            } for r in results], f, indent=2)

        logger.info(f"Pipeline execution completed. {len(results)} total operations performed.")

    asyncio.run(main())