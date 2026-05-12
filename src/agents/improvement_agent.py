import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import config
from src.utils.logger import get_logger
from src.utils.helpers import extract_json

logger = get_logger(__name__)


class ImprovementAgent:
    def __init__(self):
        agent_cfg = config['agents']['improver']
        self.llm = ChatGoogleGenerativeAI(
            model=agent_cfg['name'],
            temperature=agent_cfg['temperature'],
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", agent_cfg['prompt']),
            ("user", "{input}")
        ])
        self.chain = self.prompt | self.llm

    def suggest_improvements(self, json_report: dict, current_params: dict):
        logger.info("Improver Agent analyzing Evaluator recommendations...")

        input_msg = f"""
        Evaluator Agent Diagnosis & Context:
        {json_report}

        Current Hyperparameters:
        {current_params}
        """
        response = self.chain.invoke({"input": input_msg})
        return extract_json(response.text)
