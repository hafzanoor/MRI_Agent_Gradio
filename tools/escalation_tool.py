import streamlit as st

def check_and_escalate(prediction_data: dict, threshold: float = 0.70) -> dict:
    prediction = prediction_data["top_prediction"]
    confidence = float(prediction_data["confidence"])
    flagged = confidence < threshold

    summary = {
        "status": "Flagged for human review" if flagged else "Automated prediction; review may still be needed",
        "flagged": flagged,
        "prediction": prediction,
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "threshold_set": f"{threshold * 100:.2f}%",
        "reason": (
            f"Confidence ({confidence:.2%}) is below the review threshold ({threshold:.2%})."
            if flagged else
            "Confidence meets the threshold; this does not guarantee diagnostic correctness."
        )
    }

    if flagged:
        if "escalated_cases" not in st.session_state:
            st.session_state["escalated_cases"] = []
        st.session_state["escalated_cases"].append(summary)

    return summary