import json
from datetime import datetime
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_report(report_data: dict):
    Path("reports").mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = Path(f"reports/{date_str}_report.json")

    with open(filepath, "w") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"JSON Report saved to {filepath}")


def save_markdown_report(ticker: str, agent_markdown: str, payload: dict):
    Path("reports").mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = Path(f"reports/{date_str}_report.md")

    content = f"# AI Forecast Report: {ticker} ({date_str})\n\n"
    content += "## 1. System Metrics (Holdout Set)\n"
    content += "### Initial Model Performance\n"
    content += f"- **MAE:** {payload['metrics_holdout_initial']['MAE']:.2f}\n"
    content += f"- **RMSE:** {payload['metrics_holdout_initial']['RMSE']:.2f}\n"
    content += f"- **MAPE:** {payload['metrics_holdout_initial']['MAPE']:.4f}\n\n"
    content += "### Final Model Performance\n"
    content += f"- **MAE:** {payload['metrics_holdout_final']['MAE']:.2f}\n"
    content += f"- **RMSE:** {payload['metrics_holdout_final']['RMSE']:.2f}\n"
    content += f"- **MAPE:** {payload['metrics_holdout_final']['MAPE']:.4f}\n\n"
    content += "## 2. 7-Day Point Forecast (Median)\n"
    content += f"```text\n{payload['7_day_forecast']['point']}\n```\n\n"
    content += "## 3. Agent Evaluation\n"
    content += agent_markdown + "\n\n"
    content += "## 4. Automatic Actions Taken\n"
    if payload.get('model_adjustments'):
        content += f"Model auto-retrained on the fly with adjusted parameters: `{payload['model_adjustments']}`\n\n"
    else:
        content += "Forecast trusted. No on-the-fly model retraining required.\n\n"

    queries = payload['proof_of_search'].get('queries_executed', [])
    news = payload['proof_of_search'].get('key_findings')

    content += '## 5. Proof of Search\n\n'

    if queries:
        content += '- **queries_executed**:\n'
        for q in queries:
            content += f'  - {q}\n'

    if news:
        content += f'\n- **key_findings**: {news}\n'

    content += '\n'

    with open(filepath, "w") as f:
        f.write(content)
    logger.info(f"Markdown Report saved to {filepath}")
