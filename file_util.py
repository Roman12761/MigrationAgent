import os
from typing import List

from models import FileWithContent


def read_and_describe_files(parent_folder: str) -> List[FileWithContent]:
  """Return a JSON string describing Java files with truncated contents.

             Each entry contains the relative path and the first 2000 characters of
             the file's content. This helps the LLM understand the structure and
             semantics of the code during migration. If reading a file fails, the
             content field is omitted.
             """
  files_info: List[FileWithContent] = []
  for dirpath, _, filenames in os.walk(parent_folder):
    rel_dir = os.path.relpath(dirpath, parent_folder)
    for filename in filenames:
      rel_path = os.path.join(rel_dir, filename)
      abs_path = os.path.join(parent_folder, rel_path)
      try:
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
          content = f.read(2000)
      except Exception:
        content = ""
      files_info.append(FileWithContent(path=rel_path, content=content))
  return files_info

def write_to_files(output_dir: str, files: List[FileWithContent]) -> None:
  """Write files to the specified output directory.
  """
  for file in files:
    path = file.path
    code = file.content
    if not path or code is None:
      continue
    abs_path = os.path.join(output_dir, path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, 'w', encoding='utf-8') as f:
      f.write(code)
