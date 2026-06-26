from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

summarize_prompt = PromptTemplate(
    template="You are a helpful assistant that summarizes text. Summarize the user text: {input}",
    input_variables=["input"]
)
summarize_chain = summarize_prompt | llm | StrOutputParser()

translate_prompt = PromptTemplate(
    template="You are a helpful assistant that translates English to French. Translate the user sentence: {input}",
    input_variables=["input"]
)
translate_chain = translate_prompt | llm | StrOutputParser()

full_chain = summarize_chain | (lambda x: {"input": x}) | translate_chain

while True:
  sentence = input("What sentence would you like to translate? (Enter 'quit' to exit) ")
  if sentence == "quit":
      break
  print(full_chain.invoke({"input": sentence}))

# === REFLECTION ===
# What does LangChain replace compared to my Day 8-18 code?
# - PromptTemplate replaces hardcoded f-strings
# - ChatGroq wrapper replaces manual client.chat.completions.create()
# - StrOutputParser replaces response.choices[0].message.content
# - | pipe replaces manually passing output between functions

# What is the lambda doing?
# - summarize_chain outputs a plain string
# - translate_chain expects {"input": ...} dict
# - lambda x: {"input": x} bridges the gap between them

# What would be hard to do without LangChain?
# - Chaining 5+ steps would require messy manual passing of variables
# - Reusing the same prompt across different agents would mean copy-pasting

# What's next (Day 21)?
# - Replace your manual ReAct agent from Day 13 with LangChain's AgentExecutor
# - Same tools, same logic — but LangChain wires the loop for you