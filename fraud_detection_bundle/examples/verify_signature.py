"""Compare a questioned signature against a reference."""

from fraud_detection import FraudDetectionAgent, FraudInput


def main() -> None:
    agent = FraudDetectionAgent()
    report = agent.run(
        [
            FraudInput(
                path="samples/sig_questioned.png",
                kind="signature",
                reference_path="samples/sig_reference.png",
                label="wire_authorization",
            )
        ],
        context="Wire transfer authorization for $250,000 — verifying signature against KYC reference.",
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
