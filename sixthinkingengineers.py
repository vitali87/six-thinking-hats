import os
import sys
from dotenv import load_dotenv
from swarm import Swarm, Agent
from openai import OpenAI
import readline


def get_multiline_input(prompt):
    print(f"{prompt} (Enter two consecutive blank lines to finish)")
    lines = []
    empty_line_count = 0
    while True:
        try:
            line = input()
            if line.strip() == "":
                empty_line_count += 1
                if empty_line_count == 2:
                    break
            else:
                empty_line_count = 0
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines[:-1])  # Remove the last empty line


# Set up readline to use a history file
history_file = os.path.expanduser("~/.sixdeveloperperspectives_history")
try:
    readline.read_history_file(history_file)
except FileNotFoundError:
    pass

# Save history on exit
import atexit

atexit.register(readline.write_history_file, history_file)

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the Swarm client
client = Swarm()

# Define the six developer perspectives as independent agents
architecture_lens = Agent(
    name="Architecture Lens",
    instructions="You are the Architecture Lens, responsible for considering the overall structure and design patterns. You always provide the final answer based on all perspectives' input while not mentioning other perspectives' names or thoughts. You respond to the user as if you are the only AI assistant: your say is the final answer that will be returned to the user.",
)

functional_lens = Agent(
    name="Functional Lens",
    instructions="You are the Functional Lens, focused on core functionality and requirements. You should provide neutral and objective information about what the software needs to do.",
)

user_experience_lens = Agent(
    name="User Experience Lens",
    instructions="You are the User Experience Lens, expressing considerations about the end-user's perspective. You should provide insights on how intuitive and user-friendly the software is.",
)

security_lens = Agent(
    name="Security Lens",
    instructions="You are the Security Lens, identifying potential vulnerabilities and risks. You should be cautious and defensive in your thinking about what could go wrong and how to prevent it.",
)

performance_lens = Agent(
    name="Performance Lens",
    instructions="You are the Performance Lens, focusing on efficiency and speed optimization. You should be optimistic and think constructively about how to make the software perform better.",
)

maintenance_lens = Agent(
    name="Maintenance Lens",
    instructions="You are the Maintenance Lens, generating ideas about code readability, scalability, and long-term maintainability. You should think creatively and propose innovative solutions for easier updates and expansions in the future.",
)

# Define the classifier agent
classifier_agent = Agent(
    name="Classifier",
    instructions="""Analyze the user's prompt and categorize it into one of the following categories:
    1. Initial Design
    2. Feature Implementation
    3. Code Review
    4. Performance Optimization
    5. Security Audit
    6. Refactoring
    7. Bug Fixing
    8. User Interface Improvement
    Respond with only the category name.""",
)

# Define lens sequences for each category
lens_sequences = {
    "Initial Design": [
        functional_lens,
        architecture_lens,
        user_experience_lens,
        architecture_lens,
    ],
    "Feature Implementation": [
        functional_lens,
        architecture_lens,
        performance_lens,
        security_lens,
        maintenance_lens,
        architecture_lens,
    ],
    "Code Review": [
        maintenance_lens,
        performance_lens,
        security_lens,
        functional_lens,
        architecture_lens,
    ],
    "Performance Optimization": [performance_lens, functional_lens, architecture_lens],
    "Security Audit": [security_lens, functional_lens, architecture_lens],
    "Refactoring": [
        maintenance_lens,
        performance_lens,
        functional_lens,
        architecture_lens,
    ],
    "Bug Fixing": [
        functional_lens,
        security_lens,
        performance_lens,
        maintenance_lens,
        architecture_lens,
    ],
    "User Interface Improvement": [
        user_experience_lens,
        functional_lens,
        performance_lens,
        architecture_lens,
    ],
}


def classify_prompt(prompt):
    messages = [{"role": "user", "content": f"Classify this prompt: {prompt}"}]
    response = client.run(agent=classifier_agent, messages=messages)
    return response.messages[-1]["content"].strip()


def run_six_developer_perspectives(initial_topic):
    topic = initial_topic
    context = f"Initial Topic: {topic}\n\nInsights:\n"

    try:
        while True:
            # Classify the prompt
            category = classify_prompt(topic)
            print(f"\nClassified category: {category}")

            # Get the lens sequence for the category
            sequence = lens_sequences.get(category, lens_sequences["Bug Fixing"])
            print(f"Lens sequence: {', '.join([lens.name for lens in sequence])}")

            # Run through the lens sequence
            for lens in sequence:
                print(f"\n{lens.name} perspective:")
                if (
                    lens == architecture_lens
                    and sequence.index(lens) == len(sequence) - 1
                ):
                    # Final Architecture Lens thinking to summarize
                    messages = [
                        {
                            "role": "user",
                            "content": f"As the Architecture Lens, provide a summary based on all the insights:\n\n{context}",
                        }
                    ]
                else:
                    # Other lenses' thinking
                    messages = [
                        {
                            "role": "user",
                            "content": f"As the {lens.name}, analyze this topic: {topic}\n\nConsider the following context from previous rounds:\n{context}",
                        }
                    ]

                response = client.run(agent=lens, messages=messages)
                print(response.messages[-1]["content"])

                # Update the context with new insights
                context += f"\n{lens.name}: {response.messages[-1]['content']}\n"

            # Collect user input after each round
            user_input = get_multiline_input("\nUser")
            if isinstance(user_input, list):
                user_input = "".join(user_input)
            if user_input:
                context += f"\nUser Input: {user_input}\n"
                topic = (
                    user_input  # Update the topic with user input for the next round
                )

            print("\n--- Starting a new round of analysis ---\n")

    except KeyboardInterrupt:
        print("\n\nAnalysis stopped by user. Final context:")
        print(context)

    return context


# Example usage
if __name__ == "__main__":
    initial_topic = input("Enter your initial software development topic or problem: ")
    final_analysis = run_six_developer_perspectives(initial_topic)
    print("\nFinal Analysis:")
    print(final_analysis)
