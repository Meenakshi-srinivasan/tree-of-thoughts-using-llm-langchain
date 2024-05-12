from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
import os
import sys

load_dotenv()

class TreeOfThoughtsChain:
  def __init__(self, topic):
      self.topic = topic


  def build_chains(self):
      # Step 1 - Building LLM to draft solutions
      solutions_template = """
      Step 1:
      I'm facing a challenge related to {topic}. Can you generate two plans for implementing {topic}?
      """

      solutions_prompt = PromptTemplate(
          input_variables = ["topic"],
          template = solutions_template
      )
      solutions_llm  = ChatOpenAI(openai_api_key = os.getenv('OPENAI_API_KEY'), temperature = 0.1)
      solutions_chain = solutions_prompt | solutions_llm



      # Step 2 - Build LLM to analyse each solution drafted in the previous step
      analysis_template  = """
      Step 2:
      For each of the {solutions}, consider the advantages and disadvantages, ways to integrate them with existing urban infrastructure and planning strategies.
      Your output should be in the form of description of the plan, advantages, disadvantages and planning strategies in a JSON format for each solution.
      """

      analysis_prompt = PromptTemplate(input_variables = ["solutions"],
                                      template = analysis_template)
      analysis_llm = ChatOpenAI(openai_api_key = os.getenv('OPENAI_API_KEY'), temperature = 0.1)
      analysis_chain = analysis_prompt | analysis_llm


      # Step 3 -  Build LLM to evaluate and rank solutions
      evaluation_template = """
      Step 3:

      Evaluate and rank solutions based on {solutions} and {analysis} in previous steps,
      rank the plans on a scale of 10 based on safety, environmental impact, reliability and cost-effectiveness and justify the rank.
      Your output should be in the form of description about the plan, advantages, disadvantages, planning strategies,
      evaluation with reason in a JSON format for each solution.
      """
      evaluation_prompt = PromptTemplate(input_variables = ["solutions", "analysis"],
                                        template = evaluation_template)
      evaluation_llm = ChatOpenAI(openai_api_key = os.getenv('OPENAI_API_KEY'), temperature = 0.1)
      evaluation_chain = evaluation_prompt | evaluation_llm

      # build a sequential chain to pass the output of one LLM as input to other
      complete_chain = RunnablePassthrough.assign(solutions = solutions_chain)| RunnablePassthrough.assign(analysis = analysis_chain) | evaluation_chain
      response = complete_chain.invoke({"topic": self.topic})
      return response.content


if __name__ == '__main__':
  if len(sys.argv) == 2:
    topic = sys.argv[1]
    tree_of_thoughts_chain = TreeOfThoughtsChain(topic)
    response = tree_of_thoughts_chain.build_chains()
    print(response)

  else:
    print('Enter the topic to run LLM Chains...')
