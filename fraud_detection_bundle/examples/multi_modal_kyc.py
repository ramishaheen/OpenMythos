"""End-to-end KYC bundle review: ID image + statement PDF + signature pair.

Demonstrates the agent orchestrating all four modalities in one call.
"""

from fraud_detection import FraudDetectionAgent, FraudInput


def main() -> None:
    inputs = [
        FraudInput(path="samples/passport.jpg", kind="image", label="passport_front"),
        FraudInput(path="samples/statement.pdf", kind="document", label="proof_of_funds"),
        FraudInput(
            path="samples/sig_questioned.png",
            kind="signature",
            reference_path="samples/sig_reference.png",
            label="account_application",
        ),
        FraudInput(path="samples/liveness.mp4", kind="video", label="liveness_check"),
    ]
    agent = FraudDetectionAgent()
    report = agent.run(
        inputs,
        context=(
            "Onboarding bundle for high-net-worth individual. Confirm whether any "
            "media in the bundle has been falsified or manipulated."
        ),
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
