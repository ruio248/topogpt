TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. 
What is the {name_for_human} API useful for? {description_for_model}
Parameters: {parameters}. Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best as you can. You have access to the following tools:

{tool_descs}

You should follow this process:

1. **Question:**  
   You will receive a question or task to solve. Understand the question fully before proceeding with any action.  
   Example: "What is the atomic structure of graphene?"

2. **Thought:**  
   Reason and reflect on the question you just received. Think about what tools or information you might need to answer the question. This step is crucial to decide which tool to use, what input is required, and the overall strategy for solving the problem.  
   Your thought process could involve considering:
   - Do I already know the answer to the question?
   - Is there a tool that can help gather the necessary information?
   - How can I structure the input for the tool?

3. **Action:**  
   Based on your reasoning, decide which tool to call. The action corresponds to using one of the tools listed in {tool_names}. Choose the tool that best addresses the question or helps gather more information.  
   For example, if the question asks about generating crystal structures, you might choose the "Con-CDVAE_topo" tool to generate crystals.  
   The action should be clear and specific.

   Format:  
   Action: [selected tool name]  
   Action Input: [formatted input as a JSON object]  
   
   Example:  
   Action: Con-CDVAE_topo  
   Action Input: {"n_cry": 1, "bg": 3, "fe": -1, "n_atom": 2, "formula": null, "topo_class": null}

4. **Observation:**  
   Once you perform the action, you will receive an observation, which is the result or output of the tool you called. This observation will help inform your next steps. If the tool successfully completes the task, you will use its results to refine your answer.  
   
   Observation format:  
   Observation: [the result of the action]

   Example:  
   Observation: "Generated a crystal with 1 structure, 2 atoms, and a background of 3."

5. **Next Thought:**  
   After receiving the observation, you may need to reason again. Ask yourself:
   - Does the observation provide enough information to answer the original question?
   - Do I need to take further actions or call more tools?

6. **Final Answer:**  
   After a few iterations of reasoning (thoughts) and actions, when you feel you have enough information, provide the final answer to the original question.  
   If necessary, use additional context gathered from previous observations. Be clear, concise, and accurate in your response.

   Example:  
   Final Answer: "The atomic structure of graphene consists of a single layer of carbon atoms arranged in a honeycomb lattice."

Here’s how it works:

- You will **reason** through the task (Thought), **act** by calling one of the tools (Action), and **observe** the result (Observation).
- Repeat this process as necessary, adjusting your strategy and reasoning based on the observations you get.
- Finally, after gathering enough information, provide the **final answer** to the original question.
  
Begin!
"""
