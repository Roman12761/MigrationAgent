import os
from typing import Any, Iterator

from dotenv import load_dotenv
from langchain.output_parsers import BooleanOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from file_util import read_and_describe_files, write_to_files
from models import FilesWithContent, \
  ValidationResult, MigrationState


class MigrationAgent:

  def __init__(self, java_project_path: str, python_project_path: str, max_validation_iteration: int = 2):
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    self.java_project_path = java_project_path
    self.python_project_path = python_project_path
    self.max_validation_iteration = max_validation_iteration
    self.boolean_parser = BooleanOutputParser()

  def run(self) -> Iterator[dict[str, Any] | Any]:
    workflow = self._create_workflow()
    graph = workflow.compile()

    migrationState = MigrationState(
        java_project_path=self.java_project_path,
        python_project_path=self.python_project_path,
        max_validation_count=self.max_validation_iteration)

    return graph.stream(migrationState)

  def _create_workflow(self):
    workflow = StateGraph(MigrationState)
    workflow.add_node("read_java_files", self._read_java_files)
    workflow.add_node("validate_java_files", self._validate_java_files)
    workflow.add_node("migrate_project", self._migrate_project)
    workflow.add_node("validate_content_is_covered",
                      self._validate_content_is_covered)
    workflow.add_node("fix_python_files", self._fix_python_files)
    workflow.add_node("write_python_files", self._write_python_files)
    workflow.add_edge(START, "read_java_files")
    workflow.add_edge("read_java_files", "validate_java_files")
    workflow.add_conditional_edges("validate_java_files",
                                   self.java_validation_passed,
                                   {"failed": END, "passed": "migrate_project"})
    workflow.add_edge("migrate_project", "validate_content_is_covered")
    workflow.add_conditional_edges("validate_content_is_covered",
                                   self.validation_passed,
                                   {"passed": "write_python_files",
                                    "failed": "fix_python_files"})
    workflow.add_edge("fix_python_files", "validate_content_is_covered")
    workflow.add_edge("write_python_files", END)
    return workflow

  def _read_java_files(self, migration_state: MigrationState) -> MigrationState:
    java_project_path = migration_state.get("java_project_path", "")
    if not java_project_path:
      raise ValueError("Java project path is not specified.")

    java_files = read_and_describe_files(java_project_path)

    migration_state["java_files"] = java_files
    return migration_state

  def _validate_java_files(self,
      migration_state: MigrationState) -> MigrationState:

    system_msg = (
      "You are a code validator that specializes in Java Spring Boot and Maven projects. You"
      "will be given a JSON object that lists the files of a project (path and content)."
      "Determine whether it is a valid Spring Boot Maven application. A valid Spring Boot"
      "Maven project must include:"
      "\n\n"
      "1) A pom.xml file that declares spring-boot-starter-parent or other Spring Boot"
      "dependencies in its <parent> or <dependencies> section\n"
      "2) A src/main/java directory containing one or more .java files\n"
      "3) At least one Java class annotated with @SpringBootApplication and a main"
      "method that calls SpringApplication.run()"
      "\n\n"
      "If all of these conditions are met, output True; otherwise output False. Do not "
      "include any explanation—your response must be exactly YES or NO."
    )
    user_msg = (
      "Below is a JSON representation of a Java project. Each entry has a path and "
      "content. Check whether this project is a valid Spring Boot Maven application "
      "according to the criteria above and respond with YES or NO (no other text):\n\n"
      "{java_files}"
    )
    prompt = ChatPromptTemplate.from_messages([
      ("system", system_msg),
      ("user", user_msg),
    ])
    chain = prompt | ChatOpenAI(model="gpt-4",
                                temperature=0) | BooleanOutputParser()
    valid_java_project = chain.invoke(
        input={"java_files": migration_state.get("java_files")})
    if valid_java_project:
      migration_state["valid_java_project"] = True
    else:
      migration_state["valid_java_project"] = False
      migration_state["migration_status"] = "Failed"
      migration_state[
        "migration_failed_reason"] = "Not a valid Java Spring Boot Maven project"
    return migration_state

  def _migrate_project(self, migration_state: MigrationState) -> MigrationState:
    migration_state["migration_status"] = "In Progress"

    system_msg = (
      "You are a code migration assistant. Your task is to rewrite a Java "
      "Spring Boot application into an equivalent Python FastAPI application. "
      "Preserve **all functionality and behavior** of the original system. "
      "The Java project may have multiple layers (controllers, services, models, etc.); "
      "it's acceptable to reorganize code in Python as long as the overall logic and features remain the same. "
      "Ensure that **all controllers (endpoints)**, business logic, data models, scheduled tasks, validation rules, "
      "and security mechanisms from the Java app are fully implemented in the FastAPI version. "
      "If the Java app uses a database or JPA, set up an appropriate database layer in the FastAPI app (for example, use SQLAlchemy with a SQLite database for local usage, or an equivalent configuration) so the application can run without external dependencies. "
      "Also include any supporting files needed to run the project (e.g., a requirements.txt for Python dependencies). "
      "Your output should be a complete, runnable FastAPI project that an end-user could run on the first attempt. "
      "Do not omit any necessary files or code. If the Java project includes tests or documentation, include corresponding tests (e.g., pytest) and update documentation (README, comments) to reflect the Python implementation."
    )
    user_msg = (
      "Java project files with content:"
      "\n{java_files}\n\n"
      "Produce a parsable valid JSON document with a key 'files'. 'files' should be a list of objects, "
      "Ensure the code is **complete and syntactically correct**, with all required imports and definitions. "
      "Only include files relevant to the FastAPI project (Python code, configuration, tests, docs)."
    )
    prompt = ChatPromptTemplate.from_messages([
      ("system", system_msg),
      ("user", user_msg),
    ])

    parser = PydanticOutputParser(pydantic_object=FilesWithContent)
    chain = prompt | ChatOpenAI(model="gpt-5") | parser

    result = chain.invoke(
        input={"java_files": migration_state.get("java_files")})
    migration_state["python_files"] = result.files
    migration_state["migration_status"] = "Initially Migrated"
    return migration_state

  def _validate_content_is_covered(self,
      migration_state: MigrationState) -> MigrationState:
    migration_state["migration_status"] = "Validation"
    migration_state["current_validation_count"] = migration_state.get(
        "current_validation_count", 0) + 1

    system_prompt = (
      "You are an experienced software migration auditor.\n"
      "Your role is to determine whether a Python FastAPI project fully preserves "
      "the functional behavior and architecture of the original Java Spring Boot application.\n"
      "Apply practical, common-sense validation — do not flag stylistic or minor implementation differences.\n"
      "The migration may restructure code across different files or modules; this is acceptable as long as the overall behavior, logic, and feature set remain **equivalent**.\n"
      "Evaluate whether:\n"
      "- The application can run without errors or missing dependencies (all necessary imports, configs, and setups are present, e.g., database initialization is handled so the app starts up correctly).\n"
      "- All controllers (endpoints), services (business logic), data models, scheduled jobs, validation rules, and security mechanisms from the Java project are present and behave equivalently in the FastAPI version.\n"
      "- The dependencies from the Java Maven project have been properly replaced or adapted in Python (for example, database connections via Spring Data should be mirrored with a Python ORM or database client, scheduled tasks via @Scheduled should use an equivalent scheduler, etc.).\n"
      "- The FastAPI project includes tests that effectively verify the migrated functionality and key edge cases (if tests were expected or included in migration).\n"
      "- Documentation (e.g., README and relevant comments) has been updated to reflect the Python implementation and usage.\n"
      "Judge equivalence from an end-user and system integrator perspective: would the system function the same under typical operations?\n"
      "Do NOT suggest minor code cleanups, stylistic tweaks, or micro-optimizations.\n"
      "Take into account previous validation suggestions (if any) to be consistent in this iteration.\n\n"
      "Produce a parsable JSON object with keys 'validation_status', 'validation_fix_instructions', and 'manual_fix_suggestion'.\n"
      "- 'validation_status': a boolean that is True if the Python project fully covers the Java functionality, False otherwise.\n"
      "- 'validation_fix_instructions': an array of strings, where each string is a specific instruction to fix a gap if the migration isn't equivalent. This should focus on functional gaps (e.g., \"Implement user authentication flow in FastAPI as in the Java app\").\n"
      "- 'manual_fix_suggestion': a single string providing actionable advice for the user to manually fix the migration if this is the **final** validation attempt and it still fails. If no fixes are needed (or if not the final attempt), use an empty string here.\n"
      "Ensure the response is only the JSON with these keys and nothing else."
    )

    user_prompt = (
      "Below is a JSON representation of a Java project and its migrated Python FastAPI project. "
      "Each entry has a path and content.\n\n"
      "{java_files}\n\n"
      "{python_files}\n\n"
      "Final validation: {final_validation}\n"
      "Previous validation fix instructions (if any): {previous_instructions}\n\n"
      "Based on the criteria above, respond with the JSON object describing the validation result."
    )

    final_validation = migration_state.get("current_validation_count", 0) >= migration_state.get("max_validation_count", 2)
    if final_validation:
      return migration_state
    prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("user", user_prompt),
    ])

    parser = PydanticOutputParser(pydantic_object=ValidationResult)
    chain = prompt | ChatOpenAI(model="gpt-5") | parser

    result = chain.invoke(
        input={"java_files": migration_state.get("java_files"),
               "python_files": migration_state.get("python_files"),
               "final_validation": final_validation,
               "previous_instructions": migration_state.get(
                   "validation_fix_instructions", [])},
    )

    migration_state["validation_status"] = result.validation_status
    migration_state[
      "validation_fix_instructions"] = result.validation_fix_instructions
    if final_validation and not result.validation_status:
      migration_state["migration_status"] = "Partially Completed"
      output_messages = migration_state.get("user_output_messages", [str])
      output_messages.append(result.manual_fix_suggestion)
      migration_state["user_output_messages"] = output_messages
    return migration_state

  def _fix_python_files(self,
      migration_state: MigrationState) -> MigrationState:
    migration_state["migration_status"] = "Applying Fixes"

    system_msg = (
      "You are a code migration assistant specializing in fixing Python FastAPI applications. "
      "Your task is to apply specific fix instructions to an existing Python FastAPI project. "
      "You will be given:\n"
      "1) The current Python project files with their contents\n"
      "2) A list of validation fix instructions that describe what needs to be corrected or added\n"
      "Apply all the fix instructions to produce an updated Python project. "
      "You must provide the complete updated files, not just the changes. "
      "Ensure that:\n"
      "- All fix instructions are addressed\n"
      "- Existing functionality is preserved\n"
      "- Code follows Python and FastAPI best practices\n"
      "- All necessary imports and dependencies are included\n"
      "- The code is properly formatted and organized"
    )

    user_msg = (
      "Current Python project files:\n{python_files}\n\n"
      "Fix instructions to apply:\n{fix_instructions}\n\n"
      "Produce a parsable valid JSON document with a key 'files'. 'files' should be a list of objects, "
      "each with 'path' (relative to project root) and 'content' (the updated Python file content). "
      "Include all files (both modified and unmodified) needed for a complete FastAPI project."
    )

    prompt = ChatPromptTemplate.from_messages([
      ("system", system_msg),
      ("user", user_msg),
    ])

    parser = PydanticOutputParser(pydantic_object=FilesWithContent)
    chain = prompt | ChatOpenAI(model="gpt-5") | parser

    result = chain.invoke(
        input={
          "python_files": migration_state.get("python_files"),
          "fix_instructions": migration_state.get(
              "validation_fix_instructions")
        })

    migration_state["python_files"] = result.files
    migration_state["migration_status"] = "Fixes Applied"
    return migration_state

  def _write_python_files(self,
      migration_state: MigrationState) -> MigrationState:
    python_project_path = migration_state.get("python_project_path", "")

    python_files = migration_state.get("python_files", [])
    write_to_files(python_project_path, python_files)
    migration_state["migration_status"] = "Files Written"
    return migration_state

  def java_validation_passed(self, migration_state: MigrationState) -> str:
    if not migration_state.get("valid_java_project", False):
      return "failed"
    return "passed"

  def validation_passed(self, migration_state: MigrationState) -> str:
    if migration_state.get("validation_status", False):
      return "passed"
    else:
      if migration_state.get("current_validation_count", 0) >= migration_state.get("max_validation_count", 2):
        migration_state["migration_status"] = "Partially Completed"
        return "passed"
      else:
        return "failed"
