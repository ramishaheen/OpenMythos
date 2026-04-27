"""Run the agent on a document (PDF or image of a document)."""

from fraud_detection import FraudDetectionAgent, FraudInput


def main() -> None:
    agent = FraudDetectionAgent()
    report = agent.run(
        [FraudInput(path="samples/statement.pdf", kind="document", label="bank_statement")],
        context="KYC review for new corporate account opening.",
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
