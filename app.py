import os
import glob

import gradio as gr

try:  # ZeroGPU hardware needs a @spaces.GPU function; on CPU hardware this is a no-op
    import spaces
    gpu_task = spaces.GPU(duration=60)
except Exception:
    def gpu_task(fn):
        return fn

from tools.classifier_tool import predict_mri
from tools.gradcam_tool import generate_gradcam
from tools.escalation_tool import check_and_escalate
from tools.knowledge_tool import query_knowledge_base

GROQ_MODEL = "openai/gpt-oss-120b"

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

INTRO = """
# 🧠 Explainable Brain Tumor MRI Classification with Uncertainty-Aware Decision Support

Upload a 2D brain MRI slice. The agent classifies it into one of four classes (glioma, meningioma,
pituitary tumor, no tumor), shows a **Grad-CAM** heatmap of the regions that drove the prediction,
flags low-confidence results for human review, and writes a short report grounded in a small
knowledge base retrieved with **MiniLM embeddings + FAISS**.

**Model:** MobileNetV2 (ImageNet pre-trained, fine-tuned with PyTorch) &nbsp;·&nbsp;
**Author:** Hafza Noor · Bahauddin Zakariya University, Multan
"""

DISCLAIMER = (
    "⚠️ **For research and educational use only. Not for clinical diagnosis or patient care.** "
    "The model can be wrong, and it is especially unreliable on scans that differ from its training data "
    "(other scanners, sequences or views)."
)


def _get_llm():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(groq_api_key=key, model_name=GROQ_MODEL)
    except Exception:
        return None


def _fallback_report(pred, context, threshold):
    name = DISPLAY_NAMES.get(pred["top_prediction"], pred["top_prediction"])
    conf = pred["confidence"] * 100
    lines = ["### Summary", f"**Predicted class:** {name} ({conf:.1f}% confidence)", ""]
    if pred["confidence"] < threshold:
        lines.append(f"This confidence is below the review threshold of {threshold*100:.0f}%, "
                     "so the result is uncertain and should not be relied on.")
        lines.append("")
    lines += ["**Background:**", context, "", "_This report is for educational research only._"]
    return "\n".join(lines)


def _report(pred, context, threshold):
    llm = _get_llm()
    if llm is None:
        return _fallback_report(pred, context, threshold)
    name = DISPLAY_NAMES.get(pred["top_prediction"], pred["top_prediction"])
    prompt = (
        "You are an AI medical communicator. Write a short, educational markdown summary of the findings below.\n"
        f"Prediction: {name} ({pred['confidence']*100:.2f}% confidence).\n"
        f"Class probabilities: {pred['all_probabilities']}\n"
        f"Medical context: {context}\n"
        "Rules: Use only the information given. Do not speculate about software errors. "
        f"If confidence is below {threshold*100:.0f}%, say clearly that the result is uncertain and should not be relied on. "
        "Remind the reader this is for educational research only."
    )
    try:
        return llm.invoke(prompt).content
    except Exception:
        return _fallback_report(pred, context, threshold)


def _queue_markdown(queue):
    if not queue:
        return "No cases pending review."
    rows = ["| # | Prediction | Confidence | Reason |", "|---|---|---|---|"]
    for i, c in enumerate(queue, 1):
        rows.append(f"| {i} | {DISPLAY_NAMES.get(c['prediction'], c['prediction'])} | "
                    f"{c['confidence_percentage']} | {c['reason']} |")
    return "\n".join(rows)


@gpu_task
def analyze(image, threshold, queue):
    queue = list(queue or [])
    if image is None:
        raise gr.Error("Please upload an MRI image first.")

    pred = predict_mri(image)
    heatmap = generate_gradcam(image)
    esc = check_and_escalate(pred, threshold=threshold)
    if esc["flagged"]:
        queue.append(esc)

    context = query_knowledge_base(DISPLAY_NAMES.get(pred["top_prediction"], pred["top_prediction"]))
    report = _report(pred, context, threshold)

    probs = {DISPLAY_NAMES.get(k, k): v for k, v in pred["all_probabilities"].items()}
    status = (f"🚨 **Flagged for human review.** {esc['reason']}" if esc["flagged"]
              else f"✅ **Confidence threshold satisfied.** {esc['reason']}")

    return probs, heatmap, status, report, queue, _queue_markdown(queue), pred


def follow_up(question, pred):
    if not question or not question.strip():
        return "Please type a question."
    if not pred:
        return "Please run a scan analysis first."
    llm = _get_llm()
    if llm is None:
        return "The follow-up chat needs a GROQ_API_KEY secret, which is not set on this deployment."
    name = DISPLAY_NAMES.get(pred["top_prediction"], pred["top_prediction"])
    prompt = (
        f"Context: an educational brain MRI classifier predicted '{name}' with "
        f"{pred['confidence']*100:.1f}% confidence.\n"
        f"User question: {question}\n"
        "Give a concise, educational answer. Do not give medical advice or a diagnosis."
    )
    try:
        return llm.invoke(prompt).content
    except Exception:
        return "The language model could not be reached. Please try again later."


def _example_images():
    files = []
    for ext in ("jpg", "jpeg", "png"):
        files += glob.glob(os.path.join("examples", f"*.{ext}"))
    return sorted(files)


with gr.Blocks(title="Explainable Brain Tumor MRI Classifier") as demo:
    gr.Markdown(INTRO)
    gr.Markdown(DISCLAIMER)

    queue_state = gr.State([])
    pred_state = gr.State(None)

    with gr.Tab("🔬 Analyze Scan"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Input MRI slice", sources=["upload"], height=320)
                threshold_in = gr.Slider(0.50, 0.95, value=0.70, step=0.05,
                                         label="Review threshold: flag predictions below this confidence")
                run_btn = gr.Button("🚀 Run Analysis", variant="primary")
                examples = _example_images()
                if examples:
                    gr.Examples(examples=examples, inputs=image_in, label="Sample scans (click to try)")
            with gr.Column():
                heatmap_out = gr.Image(type="pil", label="Model attention (Grad-CAM)", height=320)
                label_out = gr.Label(num_top_classes=4, label="Predicted class probabilities")
        status_out = gr.Markdown()
        gr.Markdown("### 📝 Generated Report")
        report_out = gr.Markdown()

    with gr.Tab("💬 Ask a Question"):
        question_in = gr.Textbox(label="Ask about the latest result", placeholder="e.g. What is a meningioma?")
        ask_btn = gr.Button("Ask")
        answer_out = gr.Markdown()

    with gr.Tab("🚩 Human Review Queue"):
        queue_out = gr.Markdown("No cases pending review.")

    with gr.Tab("📊 Methods & Results"):
        gr.Markdown(
            "**Pipeline:** MobileNetV2 transfer learning → softmax probabilities → Grad-CAM "
            "(last convolutional block) → confidence-based escalation → retrieval with MiniLM embeddings "
            "and FAISS → report generation with an LLM (Groq).\n\n"
            "**Evaluation (1,600 held-out test images):** accuracy 95.2%, macro F1 0.951. "
            "A leakage check found 582 test images (36.4%) that are pixel-level duplicates of training images; "
            "on the 1,018 clean test images accuracy is 92.8% (macro F1 0.892). Glioma is the weakest class "
            "(recall 82%). Expected Calibration Error rises from 0.028 to 0.045 on clean images, and temperature "
            "scaling (T = 1.83) reduces negative log-likelihood by 15%.\n\n"
            "**Known limitations:**\n"
            "- Works on single 2D slices, not full 3D volumes.\n"
            "- On out-of-distribution scans (different scanners, sequences or views) the model can produce "
            "low-confidence misclassifications and skewed attention maps.\n"
            "- Softmax confidence is not calibrated, so a high percentage does not guarantee a correct prediction.\n"
            "- Grad-CAM shows where the model looked, not a clinical tumor segmentation."
        )

    run_btn.click(
        analyze,
        inputs=[image_in, threshold_in, queue_state],
        outputs=[label_out, heatmap_out, status_out, report_out, queue_state, queue_out, pred_state],
    )
    ask_btn.click(follow_up, inputs=[question_in, pred_state], outputs=answer_out)
    question_in.submit(follow_up, inputs=[question_in, pred_state], outputs=answer_out)


if __name__ == "__main__":
    demo.launch()
