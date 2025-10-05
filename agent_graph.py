import os
from typing import Any, Iterator

from dotenv import load_dotenv
from langchain.output_parsers import BooleanOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, List

from FileUtil import read_and_describe_files, write_to_files
from models import FileWithContent, FilesWithContent, \
  ValidationResult, MigrationState


class MigrationAgent:
  def __init__(self, java_project_path: str, python_project_path: str, max_validation_iteration:int=3):
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    self.java_project_path = java_project_path
    self.python_project_path = python_project_path
    self.max_validation_iteration = max_validation_iteration
    self.gpt_4_llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    self.gpt_5_llm = ChatOpenAI(model="gpt-5", temperature=0.0)
    self.boolean_parser = BooleanOutputParser()

  def run(self) -> Iterator[dict[str, Any] | Any]:
    workflow = StateGraph(MigrationState)
    workflow.add_node("read_java_files", self.read_java_files)
    workflow.add_node("validate_java_files", self.validate_java_files)
    workflow.add_node("migrate_project", self.migrate_project)
    workflow.add_node("validate_content_is_covered",
                      self.validate_content_is_covered)
    workflow.add_node("fix_python_files", self.fix_python_files)
    workflow.add_node("write_python_files", self.write_python_files)

    workflow.add_edge(START, "read_java_files")
    workflow.add_edge("read_java_files", "validate_java_files")
    workflow.add_conditional_edges("validate_java_files",
                                   self.java_validation_passed,
                                   {"end": END, "proceed": "migrate_project"})
    workflow.add_edge("migrate_project", "validate_content_is_covered")
    workflow.add_conditional_edges("validate_content_is_covered",
                                   self.validation_passed, {"end": "write_python_files",
                                                            "apply_fixes": "fix_python_files"})
    workflow.add_edge("fix_python_files", "validate_content_is_covered")
    workflow.add_edge("write_python_files", END)
    graph = workflow.compile()

    migrationState = MigrationState(
        java_project_path=self.java_project_path,
        python_project_path=self.python_project_path,
        max_validation_count=self.max_validation_iteration)

    return graph.stream(migrationState)


  def read_java_files(self, migration_state: MigrationState) -> MigrationState:
    """Read Java files from the specified project directory."""
    java_project_path = migration_state.get("java_project_path", "")
    if not java_project_path:
      raise ValueError("Java project path is not specified.")

    # Placeholder for actual file reading logic
    java_files = read_and_describe_files(java_project_path)

    migration_state["java_files"] = java_files
    return migration_state

  def validate_java_files(self,
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
    chain = prompt | self.gpt_4_llm | self.boolean_parser
    valid_java_project = chain.invoke(
        input={"java_files": migration_state.get("java_files")})
    if (valid_java_project):
      migration_state["valid_java_project"] = True
    else:
      migration_state["valid_java_project"] = False
      migration_state["migration_status"] = "Failed"
      migration_state[
        "migration_failed_reason"] = "Not a valid Java Spring Boot Maven project"
    return migration_state

  def migrate_project(self, migration_state: MigrationState) -> MigrationState:
    migration_state["migration_status"] = "In Progress"

    system_msg = (
      "You are a code migration assistant. Your task is to rewrite a Java "
      "Spring Boot application into a Python FastAPI application. You will be "
      "given a list of Java source files with their contents. Generate an "
      "equivalent Python project. For each output file, provide the relative path "
      "under the project root and the full content of the file. Do not omit necessary files."
    )
    user_msg = (
      "Java project files with content:"
      "\n{java_files}\n\n"
      "Produce a parsable valid JSON document with a key 'files'. 'files' should be a list of objects, "
      "each with 'path' (relative to project root) and 'content' (the Python file content). "
      "Only include Python and configuration files relevant to a FastAPI project."
    )
    prompt = ChatPromptTemplate.from_messages([
      ("system", system_msg),
      ("user", user_msg),
    ])

    parser = PydanticOutputParser(pydantic_object=FilesWithContent)
    chain = prompt | self.gpt_5_llm | parser

    result = chain.invoke(
        input={"java_files": migration_state.get("java_files")})
    migration_state["python_files"] = result.files
    migration_state["migration_status"] = "Initially Migrated"
    return migration_state

  def validate_content_is_covered(self,
      migration_state: MigrationState) -> MigrationState:
    migration_state["migration_status"] = "Validation"
    migration_state["current_validation_count"] = migration_state.get(
        "current_validation_count", 0) + 1

    system_prompt = (
      "You are an experienced software migration auditor.\n"
      "Your role is to determine whether a Python FastAPI project fully preserves "
      "the functional behavior and architecture of the original Java Spring Boot (Maven) application.\n"
      "Apply practical, common-sense validation—do not flag stylistic or minor implementation differences.\n"
      "The migration may restructure code across different files or modules; "
      "this is acceptable as long as the overall behavior, logic, and feature set remain equivalent.\n"
      "Evaluate whether:\n"
      "- Application can run without errors or missing dependencies.\n"
      "- All controllers, services, data models, scheduled jobs, validation rules, "
      "and security mechanisms from the Java project are present and behave equivalently in the FastAPI version.\n"
      "- The dependencies from the Java Maven project have been properly replaced, reimplemented, or adapted in Python.\n"
      "- The FastAPI project includes tests that effectively verify the migrated functionality and key edge cases.\n"
      "- Documentation (e.g., README, comments) has been updated to reflect the Python implementation and usage.\n"
      "Judge equivalence from an end-user and system-integrator perspective: would the system function the same under typical operations?\n"
      "Do NOT suggest minor code cleanups, stylistic tweaks, or micro-optimizations.\n"
      
      "Produce a parsable valid JSON document with a keys 'validation_status', 'validation_fix_instructions' and 'manual_fix_suggestion'"
      "'validation_status' is a boolean field, True if the Python project fully covers the Java functionality, False otherwise. "
      "'validation_six_instructions' is a list of strings, each string is a specific instruction to fix gaps in the migration. "
      "'manual_fix_suggestion' string field that provides actionable advice for the user to manually fix the migration if this is the final validation iteration and it is not successful, empty string if no fixes needed"
      "Only include Python and configuration files relevant to a FastAPI project."
    )

    user_prompt = (
      "Below is a JSON representation of a Java project and its migrated Python FastAPI project. "
      "Each entry has a path and content. Check whether the Python project fully covers the functionality "
      "of the Java project according to the criteria above and respond with a JSON object in the specified format:\n\n"
      "{java_files}\n\n"
      "{python_files}\n\n"
      "Final validation: {final_validation}"
    )

    final_validation = migration_state.get("current_validation_count",
                                           0) >= migration_state.get(
        "max_validation_count", 3)
    prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("user", user_prompt),
    ])

    parser = PydanticOutputParser(pydantic_object=ValidationResult)
    chain = prompt | self.gpt_5_llm | parser

    result = chain.invoke(
        input={"java_files": migration_state.get("java_files"),
               "python_files": migration_state.get("python_files"),
               "final_validation": final_validation},
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

  def fix_python_files(self, migration_state: MigrationState) -> MigrationState:
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
    chain = prompt | self.gpt_4_llm | parser

    result = chain.invoke(
        input={
          "python_files": migration_state.get("python_files"),
          "fix_instructions": migration_state.get(
              "validation_fix_instructions")
        })

    migration_state["python_files"] = result.files
    migration_state["migration_status"] = "Fixes Applied"
    return migration_state

  def write_python_files(self, migration_state: MigrationState) -> MigrationState:
    """Write Python files to the specified output directory."""
    python_project_path = migration_state.get("python_project_path", "")

    python_files = migration_state.get("python_files", [])
    write_to_files(python_project_path, python_files)
    migration_state["migration_status"] = "Files Written"
    return migration_state

  def java_validation_passed(self, migration_state: MigrationState) -> str:
    if not migration_state.get("valid_java_project", False):
      return "end"
    return "proceed"

  def validation_passed(self, migration_state: MigrationState) -> str:
    if migration_state.get("validation_status", False):
      return "end"
    else:
      if migration_state.get("current_validation_count",
                             0) >= migration_state.get("max_validation_count",
                                                       3):
        migration_state["migration_status"] = "Partially Completed"
        return "end"
      else:
        return "apply_fixes"
