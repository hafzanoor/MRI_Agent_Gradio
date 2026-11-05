def check_and_escalate(prediction_data: dict, threshold: float = 0.70) -> dict:
    prediction = prediction_data["top_prediction"]
    confidence = float(prediction_data["confidence"])
    flagged = confidence < threshold

    return {
        "status": "Flagged for human review" if flagged else "Automated prediction; review may still be needed",
        "flagged": flagged,
        "prediction": prediction,
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "threshold_set": f"{threshold * 100:.2f}%",
        "reason": (
            f"Confidence ({confidence:.2%}) is below the review threshold ({threshold:.2%})."
            if flagged else
            "Confidence meets the threshold; this does not guarantee diagnostic correctness."
        ),
    }
