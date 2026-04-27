"""Run the agent on a video file (e.g. liveness selfie clip)."""

from fraud_detection import FraudDetectionAgent, FraudInput


def main() -> None:
    agent = FraudDetectionAgent()
    report = agent.run(
        [FraudInput(path="samples/liveness.mp4", kind="video", label="liveness_check")],
        context="Liveness verification for remote account opening.",
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
