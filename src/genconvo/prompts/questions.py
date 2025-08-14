"""
    Prompts for the GenConvoBench dataset. https://arxiv.org/pdf/2506.06266

    Question Generation: A series of distinct prompt templates (detailed below), designed to elicit different
reasoning traces (e.g., factual recall, synthesis, multi-hop reasoning), are used to generate questions. For
the given document and each prompt template, we ask the model to generate 16 unique questions. This
involves providing the model with the full document content alongside the specific question-generation
prompt.

    Answer Generation: Subsequently, for each generated question, Claude Sonnet 3.7 is prompted again
with the original full document and the generated question to produce an answer. This process ensures
that the answers are grounded in the provided document.
"""

FACTUAL_PROMPT = """
    Please generate a question to test someone's ability to remember factual details from the document. 
    
    The answer should be a few tokens long and be a factual detail from the statement, such as a number, entity,
    date, title, or name.

    This question should not be common knowledge: instead, it should be something that is only answerable
    via information in the document.
"""

KNOWLEDGE_PROMPT = """
    Please generate a question that requires combining information mentioned both inside and outside the
    document.

    This question should require using a fact from the document and also a fact that you are confident about,
    but is not mentioned in the document. For instance: 
    
    - What are the founding dates of the companies that got acquired this year? 
    This is a good question because the names of the acquired companies are mentioned in the document and the founding dates are not mentioned. 
    
    - What is the name of the CEO's spouse? 
    This is a good question because the name of the CEO is mentioned in the document and the spouse's name is not mentioned.

    The answer should be a fact that is a few tokens long such as a number, entity, date, title, or name.
"""

DISJOINT_PROMPT = """
    Please generate a multi-hop question that tests someone's ability to use factual information mentioned
    in at least two very different sub-sections of the document.

    This question shouldn't be a standard question about this kind of document. Instead, it should ask
    about two particularly disconnected ideas, like comparing information about the amount of owned space
    for the company headquarters with the amount of dollars of estimated liability or comparing the revenue
    number with the number of employees.

    This question should also test one's ability to do retrieval: do not give away part of the answer in
    the question. Ensure that for one to get the correct answer to the question, they need to understand
    the document.

    The answer should be a short: for example, a number, entity, date, title, or name.
"""

SYNTHESIZED_PROMPT = """
    Please generate a question that requires synthesizing and aggregating information in the document.

    For instance, you could ask someone to summarize a page of the document, list all the key competitors
    mentioned in the document, or summarize the company's business model.
"""

STRUCTURED_PROMPT = """
    Please generate a question that requires understanding the structure of the document.

    This question should be more about the structure of the document, rather than the precise statement
    details. For instance, you could ask someone to list the titles of all the sections in the document,
    describe the document structure, report the total number of pages, ask which section amongst two sections
    comes first, or report the section with the largest number of tables.
"""

CREATIVE_PROMPT = """
    Please generate a question about the document to test someone's ability to comprehend the content of the
    document. This question specifically should be focused on their ability to generalize the information
    about the document to a strange question of sorts.

    This question shouldn't be a standard question about this kind of document, it should ask to do something
    abnormal and creative. For example, you could ask someone to write a short poem about the document.
"""

COUNTING_PROMPT = """
    Please generate a question that requires counting how frequently different events occur in the document.

    This question should be about statistical properties of the document, rather than the statement details.
    For instance, you could ask someone to count the number of times the word "million" is mentioned or
    count the length of the shortest section title.

    The answer should be a number.
"""


REASONING_PROMPT = """
    Please generate a question that requires mathematical reasoning over the values in the document.

    This question should require going beyond the facts directly mentioned in the statement, such as asking
    to compute the percentage increase in revenue between two years, find the largest expense category, or
    calculate difference in profit between two years. You may be asked to generate many questions,
    so make sure they are diverse and not too similar to each other.

    The answer should be a number.
"""

FACTUAL_V2_PROMPT = """
    Generate a factual recall question about a specific entity, date, or name from the document.

    Format: "Who/What/When [specific question]?"
    Answer: Must be an exact entity name, date, or proper noun from the document (2-4 words max).

    The answer should be unambiguous and directly stated in the document.
"""

REASONING_V2_PROMPT = """
    Generate a mathematical reasoning question requiring calculation over document values.

    Format: "What is the [percentage/ratio/difference] of [specific calculation]?"
    Answer: Must be a precise number with units (e.g., "12.5%", "$2.3M", "1.8x").

    Question should require combining 2+ values from different parts of the document.
"""

COUNTING_V2_PROMPT = """
    Generate a counting question about document structure or content frequency.

    Format: "How many [items] are [condition]?"
    Answer: Must be a single integer (e.g., "7", "23").

    Focus on countable elements like sections, tables, mentions of specific terms, or occurrences.
"""

SYNTHESIS_V2_PROMPT = """
    Generate a multiple choice question testing document comprehension.

    Format: Question with 5 options (A/B/C/D/E).
    Answer: Single letter (A, B, C, D, or E). E always means "There is not enough information to answer the question".  

    Question should require understanding main themes, risks, or business model elements.
"""

GEN_CONVO_PROMPT_REGISTRY = {
    "factual": FACTUAL_PROMPT,  # Used in the paper
    "knowledge": KNOWLEDGE_PROMPT,
    "disjoint": DISJOINT_PROMPT,  # Used in the paper
    "synthesized": SYNTHESIZED_PROMPT,  # Used in the paper
    "structured": STRUCTURED_PROMPT,  # Used in the paper
    "creative": CREATIVE_PROMPT,  # Used in the paper
    "counting": COUNTING_PROMPT,
    "reasoning": REASONING_PROMPT,  # Used in the paper

    "factual_v2": FACTUAL_V2_PROMPT, # Mauri's prompts
    "reasoning_v2": REASONING_V2_PROMPT, # Mauri's prompts
    "counting_v2": COUNTING_V2_PROMPT, # Mauri's prompts
    "synthesis_v2": SYNTHESIS_V2_PROMPT, # Mauri's prompts
}
