import random
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import v2

from dataset import CLASSES, build_splits
from train_transfer import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model

CHECKPOINT = Path("transfer_model.pt")

device = torch.accelerator.current_accelerator() or torch.device("cpu")
model = build_model(weights=None).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
model.eval()

_, _, test_samples = build_splits()

transform = v2.Compose(
    [
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def predict_probabilities(image):
    """Return a class-name to probability dict for one PIL image."""
    inputs = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(inputs)
    probabilities = torch.softmax(logits[0], dim=0).cpu().tolist()
    return dict(zip(CLASSES, probabilities))


def new_round(score):
    """Pick a random test frame and reset the guess panel."""
    path, true_label = random.choice(test_samples)
    score = {**score, "path": path, "true_label": true_label}
    return (
        Image.open(path).convert("RGB"),
        None,
        "",
        None,
        gr.update(interactive=True),
        score,
    )


def submit_guess(guess, score):
    """Reveal the true movie and the model's prediction for the current frame."""
    if guess is None:
        return "Pick a movie first.", None, gr.update(interactive=True), score

    path = score["path"]
    true_name = CLASSES[score["true_label"]]
    probabilities = predict_probabilities(Image.open(path).convert("RGB"))
    model_name = max(probabilities, key=probabilities.get)

    score["rounds"] = score.get("rounds", 0) + 1
    score["human_correct"] = score.get("human_correct", 0) + (guess == true_name)
    score["model_correct"] = score.get("model_correct", 0) + (model_name == true_name)

    result = (
        f"True movie: {true_name}\n"
        f"Your guess: {guess} ({'correct' if guess == true_name else 'wrong'})\n"
        f"Model guess: {model_name} ({'correct' if model_name == true_name else 'wrong'})\n\n"
        f"Rounds: {score['rounds']} | "
        f"Your accuracy: {score['human_correct'] / score['rounds']:.0%} | "
        f"Model accuracy: {score['model_correct'] / score['rounds']:.0%}"
    )
    return result, probabilities, gr.update(interactive=False), score


with gr.Blocks(title="Guess the Movie") as demo:
    gr.Markdown(
        "# Guess the Movie\nLook at the frame, pick a movie, then see how the model did."
    )
    score_state = gr.State({})

    with gr.Row():
        image = gr.Image(type="pil", label="Frame", interactive=False)
        with gr.Column():
            guess = gr.Radio(CLASSES, label="Your guess")
            submit_button = gr.Button("Submit guess", variant="primary")
            next_button = gr.Button("Next frame")
            result_text = gr.Textbox(label="Result", lines=6, interactive=False)
            probabilities_label = gr.Label(
                label="Model probabilities", num_top_classes=len(CLASSES)
            )

    round_outputs = [image, guess, result_text, probabilities_label, submit_button, score_state]

    submit_button.click(
        submit_guess,
        inputs=[guess, score_state],
        outputs=[result_text, probabilities_label, submit_button, score_state],
    )
    next_button.click(new_round, inputs=[score_state], outputs=round_outputs)
    demo.load(new_round, inputs=[score_state], outputs=round_outputs)


if __name__ == "__main__":
    demo.launch()
