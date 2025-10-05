from typing import TypedDict, List

from pydantic import BaseModel


class FileWithContent(BaseModel):
    path: str
    content: str


class FilesWithContent(BaseModel):
    files: list[FileWithContent]


class ValidationResult(BaseModel):
    validation_status: bool
    validation_fix_instructions: list[str] = []
    manual_fix_suggestion: str = ""


class MigrationState(TypedDict):
    java_project_path: str
    python_project_path: str
    max_validation_count: int
    current_validation_count: int
    migration_status: str  # "Pending", "In Progress", "Completed", "
    migration_failed_reason: str
    valid_java_project: bool
    validation_status: bool
    validation_fix_instructions: List[str]
    user_output_messages: List[str]
    java_files: List[FileWithContent]
    python_files: List[FileWithContent]
