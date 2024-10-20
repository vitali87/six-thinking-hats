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
history_file = os.path.expanduser("~/.sixthinkinghats_history")
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

# Define the six thinking hats as independent agents
blue_hat = Agent(
    name="Blue Hat",
    instructions="You are the Blue Hat, responsible for managing the thinking process and synthesizing information while following the thinking process of all the hats. They help you shape your answer. You always provide the final answer based on all hats' input while not mentioning other hats' names or thoughts. You respond to the user as if you are the only AI assistant: your say is the final answer that will be returned to the user.",
)

white_hat = Agent(
    name="White Hat",
    instructions="You are the White Hat, focused on gathering facts and information. You should provide neutral and objective data without interpretation.",
)

red_hat = Agent(
    name="Red Hat",
    instructions="You are the Red Hat, expressing emotions, feelings, and intuitions. You should provide gut reactions without justification.",
)

black_hat = Agent(
    name="Black Hat",
    instructions="You are the Black Hat, identifying risks, difficulties, and potential problems. You should be cautious and defensive in your thinking.",
)

yellow_hat = Agent(
    name="Yellow Hat",
    instructions="You are the Yellow Hat, focusing on benefits and seeking harmony. You should be optimistic and think constructively about the subject.",
)

green_hat = Agent(
    name="Green Hat",
    instructions="You are the Green Hat, generating new ideas and possibilities. You should think creatively and propose innovative solutions.",
)

# Define the classifier agent
classifier_agent = Agent(
    name="Classifier",
    instructions="""Analyze the user's prompt and categorize it into one of the following categories:
    1. Initial Ideas
    2. Choosing between alternatives
    3. Identifying Solutions
    4. Quick Feedback
    5. Strategic Planning
    6. Process Improvement
    7. Solving Problems
    8. Performance Review
    Respond with only the category name.""",
)

# Define hat sequences for each category
hat_sequences = {
    "Initial Ideas": [white_hat, green_hat, blue_hat],
    "Choosing between alternatives": [
        white_hat,
        green_hat,
        yellow_hat,
        black_hat,
        red_hat,
        blue_hat,
    ],
    "Identifying Solutions": [white_hat, black_hat, green_hat, blue_hat],
    "Quick Feedback": [black_hat, green_hat, blue_hat],
    "Strategic Planning": [
        yellow_hat,
        black_hat,
        white_hat,
        blue_hat,
        green_hat,
        blue_hat,
    ],
    "Process Improvement": [
        white_hat,
        white_hat,
        yellow_hat,
        black_hat,
        green_hat,
        red_hat,
        blue_hat,
    ],
    "Solving Problems": [
        white_hat,
        green_hat,
        red_hat,
        yellow_hat,
        black_hat,
        green_hat,
        blue_hat,
    ],
    "Performance Review": [
        red_hat,
        white_hat,
        yellow_hat,
        black_hat,
        green_hat,
        blue_hat,
    ],
}


def classify_prompt(prompt):
    messages = [{"role": "user", "content": f"Classify this prompt: {prompt}"}]
    response = client.run(agent=classifier_agent, messages=messages)
    return response.messages[-1]["content"].strip()


def run_six_thinking_hats(initial_topic):
    topic = initial_topic
    context = f"Initial Topic: {topic}\n\nInsights:\n"

    try:
        while True:
            # Classify the prompt
            category = classify_prompt(topic)
            print(f"\nClassified category: {category}")

            # Get the hat sequence for the category
            sequence = hat_sequences.get(category, hat_sequences["Solving Problems"])
            print(f"Hat sequence: {', '.join([hat.name for hat in sequence])}")

            # Run through the hat sequence
            for hat in sequence:
                print(f"\n{hat.name} thinking:")
                if hat == blue_hat and sequence.index(hat) == len(sequence) - 1:
                    # Final Blue Hat thinking to summarize
                    messages = [
                        {
                            "role": "user",
                            "content": f"As the Blue Hat, provide a summary based on all the insights:\n\n{context}",
                        }
                    ]
                else:
                    # Other hats' thinking
                    messages = [
                        {
                            "role": "user",
                            "content": f"As the {hat.name}, analyze this topic: {topic}\n\nConsider the following context from previous rounds:\n{context}",
                        }
                    ]

                response = client.run(agent=hat, messages=messages)
                print(response.messages[-1]["content"])

                # Update the context with new insights
                context += f"\n{hat.name}: {response.messages[-1]['content']}\n"

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
    initial_topic = input("Enter your initial topic or problem: ")
    final_analysis = run_six_thinking_hats(initial_topic)
    print("\nFinal Analysis:")
    print(final_analysis)
