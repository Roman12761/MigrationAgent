from agent_graph import MigrationAgent

agent = MigrationAgent(
  java_project_path="/Users/rmovc/Downloads/agent_tools/chatgptagent/input",
  python_project_path="/Users/rmovc/Downloads/agent_tools/chatgptagent/output2")

events = agent.run()

for event in events:
  print(event)
  print("--------")
