INSTRUCT_GENERATE_Prompt="""
The background knowledge is:
{background_knowledge}

**Important instructions:**
- Ensure that **all questions can be answered directly from the background information** to ensure high-quality question-answer pairs for fine-tuning purposes.
- The number of generated questions should reflect the **depth and variety** of the background information**. If the content is rich, generate more questions to cover various aspects. If it is brief or focused on a few points, generate fewer questions. Do not aim for exactly 10 questions; aim for **quality over quantity**.
- Do **not** use demonstrative pronouns like "this" or "these" in any question.
- Do **not** refer to external context, such as using phrases like "in the article" ,"in the FIg", "in the experiments", or "mentioned in the text."
- Each question should be independent and reflect natural user interaction.
- Ensure the questions cover diverse aspects of the content: factual details, definitions, practical applications, evaluations, and more.

**Format:**
- Each question must be numbered sequentially (e.g., Question_1, Question_2, etc.).
- Questions must strictly follow the format below:

Example format:
'Question_1': ...
'Question_2': ...
'Question_3': ...
...

Generate a list of questions, with each on a new line within single quotes. Do not provide any explanations or additional content.
"""



Reading_Comprehension_Prompt ="""
The background knowledge is:
{background_knowledge}

Please answer the following question based on the
content of the article above:
{response}

Please answer this question as thoroughly as possible,
but do not change the key information in the original
text, and do not include expressions such as “based
on the above article” in the answer.
Please generate the corresponding answer in the following format:
Question: ...
Answer: ...
"""


Quality_Template = """
You are a helpful and precise assistant for checking the quality of instruction-tuning data for large language models. 
Your task is to evaluate the given question-answer pair using the criteria described below.
The background information for these QA pairs comes from a scientific research article in the field of topology, and it should be considered when evaluating the content. The context is as follows:

**Background Information:**  
{background_knowledge}

Please review the following question-answer pair:

**Question:**  
{question}

**Answer:**  
{answer}

Evaluate the question-answer pair based on the following criteria:

1. **Clarity**: The sample should be clear, specific, and unambiguous, providing a well-defined task for the model to perform.
2. **Complexity**: The sample should be of advanced complexity, requiring a high level of comprehension and cognitive processing, especially related to the field of topology, and should challenge the language model significantly.
3. **Correctness**: The sample is impeccably written, with flawless grammar, syntax, and structure, demonstrating exceptional clarity and professionalism, and should reflect the key concepts or findings from the article accurately.
4. **Usefulness**: The sample should be highly useful, and contribute to expanding the model’s knowledge base, especially in the context of topological materials or phenomena discussed in the article.
5. **Adaptability**: The sample should be flexible enough to apply to different contexts or use cases, showing some adaptability in scientific or educational settings related to topology.

Please also consider the relevance and accuracy of the question-answer pair in relation to the background material when evaluating the scores.

For each of the criteria, please assign a score from **1 to 5** based on the following scale:

- **1**: Very Poor — The QA pair is unclear, inaccurate, and does not contribute meaningfully to the understanding of the topic.
- **2**: Poor — The QA pair is somewhat unclear, lacks precision, and contains some inaccuracies.
- **3**: Average — The QA pair is clear and accurate but may lack depth or could be slightly imprecise.
- **4**: Good — The QA pair is clear, accurate, and reflects a good level of understanding of the topic.
- **5**: Excellent — The QA pair is very clear, highly accurate, and provides detailed insights into the topic.

**Format:**
- Each question must be numbered sequentially (e.g., Question_1, Question_2, etc.).
- Questions must strictly follow the format below:

**Example format:**                           
{{
  Clarity: 5,
  Complexity: 5,
  Correctness: 5,
  Usefulness: 5,
  Adaptability: 4,
  Total: 24
}}
"""


"""
**Example format:**                            
{
  "Clarity": 5,
  "Complexity": 5,
  "Correctness": 5,
  "Usefulness": 5,
  "Adaptability": 4,
  "Total": 24
}
"""


Quality_Template_1 = """
You are a professional expert evaluating question-answer pairs about topological materials for training large language models. Your task is to assess the suitability of the given question-answer pair based on its relevance, accuracy, depth, and clarity.

**Evaluation Criteria:**

1. **Topological Relevance (1-3):** How directly does the question-answer pair relate to core concepts in topological materials (e.g., topological insulators, superconductors, etc.)?
2. **Accuracy (1-3):** How accurate and consistent is the answer with established knowledge in topological materials?
3. **Depth (1-3):** Does the answer demonstrate a profound understanding of the field, potentially involving advanced concepts or cutting-edge research?
4. **Clarity (1-3):** How clear, understandable, and unambiguous is the answer?

**Scoring:**

* **1:** Low (Irrelevant, incorrect, overly simplistic, or unclear).
* **2:** Medium (Somewhat relevant, minor errors or inaccuracies, or moderately clear).
* **3:** High (Highly relevant, accurate, demonstrates deep understanding, and clearly expressed).

**Background Information:**
{background_knowledge}

**Question-Answer Pair:**

**Question:** {question}
**Answer:** {answer}

**Evaluation and Scoring:**

Provide a JSON object with scores for each criterion and a total score. The total score is the sum of the individual scores.

**Output Format (JSON):**

{{
  "Topological Relevance": <score>,
  "Accuracy": <score>,
  "Depth": <score>,
  "Clarity": <score>,
  "Total": <total_score>
}}
"""



QA_Generation_Template = """
Generate a single question-answer pair based on the provided background knowledge in the field of topological materials. Ensure the question is relevant and can be directly answered from the information. The answer should be clear, accurate, and concise.

**Background Information:**
{background_knowledge}

**Instructions:**
- Avoid demonstrative pronouns like "this" or "these" in the question.
- Output only the question and answer in the following JSON format:

{{
  "Question": "<Your question here>",
  "Answer": "<Your answer here>"
}}
"""

QA_COT_Template="""
Generate a single question-answer pair based on the provided background knowledge in the field of topological materials. Use a **Chain of Thought reasoning process** to systematically analyze and derive a question that goes beyond simple recall, encouraging deeper understanding. The answer should be precise, clear, and directly derived from the background information.

**Background Information:**
{background_knowledge}

**Instructions:**
1. **Step 1 - Extract Core Concepts and Facts:**
   - Carefully analyze the background information and identify the most critical concepts, facts, or relationships. These could include definitions, principles, phenomena, or applications that form the foundation of the material.

2. **Step 2 - Establish Connections and Relationships:**
   - Reflect on how the identified concepts and facts are related to each other. Look for causal relationships, dependencies, comparisons, or implications between these concepts.
   - Consider how these relationships can help formulate a meaningful question that assesses deeper understanding or the ability to apply these concepts.

3. **Step 3 - Formulate a Challenging and Insightful Question:**
   - Based on the connections and relationships identified in Step 2, design a question that tests not only the recall of facts but also the ability to synthesize and apply knowledge.
   - The question should encourage the examinee to demonstrate their understanding of how various elements of the background information come together.

4. **Step 4 - Derive a Clear and Accurate Answer:**
   - Construct an answer that is directly supported by the background information. Ensure that the answer clearly addresses the question and integrates relevant facts in a concise manner.
   - The answer should reflect both the theoretical understanding and practical applications, if applicable, based on the provided material.

**Output Format:**
Please provide the Chain of Thought reasoning and the final question-answer pair in the following JSON format:

{{
  "Question": "<Your thoughtful and challenging question here>",
  "Chain_of_Thought": "<Detailed step-by-step reasoning process that leads to the question, including connections and relationships between concepts>",
  "Answer": "<Clear, concise, and accurate answer based on the background knowledge>"
}}
"""

