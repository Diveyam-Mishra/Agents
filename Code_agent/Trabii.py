from phi.agent import Agent
from dotenv import load_dotenv
from phi.tools.crawl4ai_tools import Crawl4aiTools
from Setup  import azure_model, knowledge_base
from phi.tools.python import PythonTools
load_dotenv()
 
File_searcher = Agent(
    model=azure_model,
    knowledge_base=knowledge_base,
    search_knowledge=True,
    name="File Searcher",
    role="Given a task by the user Your task is to find relevant files from it and provide any information that might be needed from those files",
    tools=[Crawl4aiTools(max_length=None),PythonTools()],
)
Task_maker = Agent(
    model=azure_model,
    knowledge_base=knowledge_base,
    name="Task Maker",
    search_knowledge=True,
    role="Given a task by the user Your task is break down the problem by thinking through it step by step and develop multiple strategies to solve the problem.",
    tools=[Crawl4aiTools(max_length=None),PythonTools()],
)
First_agent = Agent(
    model=azure_model,
    knowledge=knowledge_base,
    search_knowledge=True,
    instructions=[
        """You are python coder who loves coding and can work on DBMS as well and you are given a task to work on a repository.
        A repository of FASTAPI backend is given to u which contains a lot of python files and U have to make new functionalities. 
        If its a new functionality name the file and write the code in it. If its a bug fix then write the bug 
        and the solution in the same file. If its a code review then write the review in the same file. 
        If its a documentation then write the documentation in the same file. If its a testing then write the test cases in the same file. 
        If its a deployment then write the deployment steps in the same file. 
        If its a configuration then write the configuration steps in the same file. 
        If its a monitoring then write the monitoring steps in the same file. 
        If its a scaling then write the scaling steps in the same file. 
        If its a security then write the security steps in the same file. If its a performance then write the performance steps in the same file. 
        If its a compliance then write the compliance steps in the same file. If its a cost then write the cost steps in the same file.""",
    ],
    Tools=[PythonTools()],   read_chat_history=True,
    show_tool_calls=True,
    markdown=True
)

Checker_reader = Agent(
    model=azure_model,
    search_knowledge=True,
    name="Checker Reader",
    role="You are given response from the first agent and you have to check the response and provide the feedback to the first agent.",
    tools=[Crawl4aiTools(max_length=None),PythonTools()],
)
from phi.tools.googlesearch import GoogleSearch

web_searcher = Agent(
    model=azure_model,
    search_knowledge=True,
    tools=[GoogleSearch(),PythonTools()],
    description=" Open Source Contributor Reading on the given topic.",
    instructions=[
        "You are a Open Source Contributor and u understand how important File Structure is for the given repo So u make sure when a change is made it accounts for all changes needed in anotehr files related to it by back tracking through import files and it is documented properly."
    ],
    show_tool_calls=True,
    debug_mode=True,
)
hn_team = Agent(
    model=azure_model,
    search_knowledge=True,
    name="Coding Team",
    team=[File_searcher,Task_maker,First_agent, web_searcher,Checker_reader],
    instructions=[
        "First - Carefully analyze the task",
        "Then, break down the problem by thinking through it step by step and develop multiple strategies to solve the problem.",
        "Then, examine the users intent develop a step by step plan to solve the problem."
        "Work through your plan step-by-step, executing any tools as needed.",
        "  4. Reasoning: Explain the logic behind this step in the first person, including:\n"
        "     - Necessity: Why this action is necessary.\n"
        "     - Considerations: Key considerations and potential challenges.\n"
        "     - Progression: How it builds upon previous steps (if applicable).\n"
        "     - Assumptions: Any assumptions made and their justifications.\n",
        "  5. Next Action: Decide on the next step:\n"
        "     - continue: If more steps are needed to reach an answer.\n"
        "     - validate: If you have reached an answer and should validate the result.\n"
        "     - final_answer: If the answer is validated and is the final answer.\n"
        "     Note: you must always validate the answer before providing the final answer.\n",
        "  6. Confidence score: A score from 0.0 to 1.0 reflecting your certainty about the action and its outcome."
        "Handling Next Actions:\n"
        "  - If next_action is continue, proceed to the next step in your analysis.\n"
        "  - If next_action is validate, validate the result and provide the final answer.\n"
        "  - If next_action is final_answer, stop reasoning."
        "Remember - If next_action is validate, you must validate your result\n"
        "  - Ensure the answer resolves the original request.\n"
        "  - Validate your result using any necessary tools or methods.\n"
        "  - If there is another method to solve the task, use that to validate the result.\n",
        "Additional Guidelines:\n"
        "  - Remember to run any tools you need to solve the problem.\n"
        "  - Take at 2-3 least steps to solve the problem.\n"
        "  - If you have all the information you need, provide the final answer.\n"
        "  - IMPORTANT: IF AT ANY TIME THE RESULT IS WRONG, RESET AND START OVER."
    ],
    tools=[Crawl4aiTools(max_length=None),PythonTools()],
    save_response_to_file='responses.txt',
    show_tool_calls=True,
    markdown=True,
)
with open("responses.txt", "a") as file:
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() == 'exit':
            break
        
        response = hn_team.print_response(user_input)
        
        file.write(f"Question: {user_input}\nResponse: {response}\n\n")
        print("Response saved to file.")
