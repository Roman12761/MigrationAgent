from agent_graph import MigrationAgent

agent = MigrationAgent(
  java_project_path="/Users/rmovc/IdeaProjects/migration-agent/input_simple_project",
  python_project_path="/Users/rmovc/IdeaProjects/migration-agent/output_simple_project")

events = agent.run()

for event in events:
  print(event)
  print("--------")
