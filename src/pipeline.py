from config.settings import config
from src.db.sqlite import SQLiteDB
from src.data.yahoo_finance import fetch_ticker_data
from src.forecasting.xgboost_model import XGBoostQuantileModel
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.improvement_agent import ImprovementAgent
from src.utils.report import save_report, save_markdown_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(ticker: str):
    db = SQLiteDB()

    try:
        logger.info("Data Ingestion....")
        raw_data = fetch_ticker_data(ticker)
        db.save_dataframe(raw_data, "prices")
        data = db.load_dataframe("prices")

        logger.info("Prediction Model...")
        base_params = config['models']['xgboost'].copy()
        model = XGBoostQuantileModel(config=base_params)
        initial_forecast = model.train_and_predict(data)
        logger.info(
            f"Holdout Set Metrics: "
            f"MAE={initial_forecast['metrics']['MAE']:.2f}, "
            f"RMSE={initial_forecast['metrics']['RMSE']:.2f}, "
            f"MAPE={initial_forecast['metrics']['MAPE']:.4f}"
        )

        logger.info("AI Agent Evaluator...")
        evaluator = EvaluatorAgent()
        eval_output = evaluator.evaluate(
            ticker,
            initial_forecast['metrics'],
            initial_forecast
        )

        proof_of_search = eval_output.get('proof_of_search', {})
        markdown_report = eval_output.get('markdown_report', "No markdown generated.")
        json_report = eval_output.get('json_report', {})

        logger.info("Feedback & Model Improvement...")
        improver = ImprovementAgent()
        adjustment_response = improver.suggest_improvements(
            json_report,
            config['models']['xgboost']
        )

        final_forecast = initial_forecast
        new_params = adjustment_response.get("new_params", None)

        if new_params and json_report.get("recommend", "").lower() in ["retrained", "adjusted"]:
            logger.info(f"Auto-adjusting model on the fly. New params: {new_params}")
            if 'window_size' not in new_params:
                new_params['window_size'] = config['models']['xgboost']['window_size']
            if 'forecast_horizon' not in new_params:
                new_params['forecast_horizon'] = config['models']['xgboost']['forecast_horizon']

            improved_model = XGBoostQuantileModel(config=new_params)
            final_forecast = improved_model.train_and_predict(data)
            logger.info(
                f"Improved Holdout MAE: "
                f"MAE={final_forecast['metrics']['MAE']:.2f}, "
                f"RMSE={final_forecast['metrics']['RMSE']:.2f}, "
                f"MAPE={final_forecast['metrics']['MAPE']:.4f}"
            )

        report_payload = {
            "ticker": ticker,
            "metrics_holdout_initial": initial_forecast['metrics'],
            "metrics_holdout_final": final_forecast['metrics'],
            "agent_evaluation": json_report,
            "model_adjustments": new_params,
            "7_day_forecast": final_forecast['forecasts'],
            "proof_of_search": proof_of_search
        }

        save_report(report_payload)
        save_markdown_report(ticker, markdown_report, report_payload)
        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.exception(
            f"Pipeline failed for ticker={ticker}. Error: {e}"
        )
    finally:
        logger.info("Pipeline finished.")
