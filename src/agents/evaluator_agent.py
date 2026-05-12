import os
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from config.settings import config
from src.utils.logger import get_logger
from src.utils.helpers import extract_json, get_current_datetime

logger = get_logger(__name__)


class EvaluatorAgent:
    def __init__(self):
        agent_cfg = config['agents']['evaluator']
        self.llm = ChatGoogleGenerativeAI(
            model=agent_cfg['name'],
            temperature=agent_cfg['temperature'],
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.tools = [
            TavilySearch(
                max_results=5,
                include_answer=True,
                search_depth="advanced",
                topic="news",
                time_range="week"
            )
        ]

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=agent_cfg['prompt'],
            middleware=[
                ModelCallLimitMiddleware(
                    thread_limit=10,
                    run_limit=5,
                    exit_behavior="end",
                )
            ],
            debug=False
        )

    def evaluate(self, ticker: str, metrics: dict, forecast_results: dict):
        logger.info("Agent executing: Evaluate 7-day trend -> Diagnose -> Context -> Recommend")

        points = forecast_results['forecasts']['point']
        ci_95_lower = forecast_results['forecasts']['95_ci_lower']
        ci_95_upper = forecast_results['forecasts']['95_ci_upper']

        forecast_table = "\n".join([
            f"Day {i+1}: ${p:.2f} (95% CI: ${l:.2f} - ${u:.2f})"
            for i, (p, l, u) in enumerate(zip(points, ci_95_lower, ci_95_upper))
        ])
        datetime_str = get_current_datetime()

        input_msg = f"""
        Current Date: {datetime_str}
        Ticker: {ticker}

        ### Holdout Set Performance
        MAE: {metrics['MAE']:.2f}
        RMSE: {metrics['RMSE']:.2f}
        MAPE: {metrics['MAPE']:.4f}

        ### 7-Day Predicted Forecast
        {forecast_table}

        Analyze the trajectory of these 7 days.
        Execute the 5 required tasks (Evaluate, Diagnose, Contextualize, Recommend, Report) and return the JSON.
        """

        response = self.agent.invoke({"messages": [("user", input_msg)]})

        return extract_json(response['messages'][-1].text)
