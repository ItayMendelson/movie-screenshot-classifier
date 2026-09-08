import random
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import v2

from dataset import CLASSES, build_splits
from train_transfer import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model

CHECKPOINT = Path("checkpoints/transfer_model.pt")

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
    gr.Markdown("MOVIE CLASSIFIER / THE FRAME CHALLENGE", elem_id="eyebrow")
    gr.Markdown(
        "# Guess the Movie\nTen films. One frame. Can you read the scene better than the model?",
        elem_id="game-heading",
    )
    score_state = gr.State({})

    with gr.Row(elem_id="game-layout"):
        with gr.Column(scale=6, min_width=320):
            image = gr.Image(
                type="pil", label="The frame", interactive=False,
                height=400, buttons=["fullscreen"], elem_id="movie-frame",
            )
            result_text = gr.Textbox(
                label="The reveal", lines=6, interactive=False,
                placeholder="Make your pick to reveal the movie and compare your scores.",
                elem_id="round-result",
            )
        with gr.Column(scale=5, min_width=320):
            guess = gr.Radio(CLASSES, label="Make your pick", elem_id="movie-choices")
            with gr.Row():
                submit_button = gr.Button("Submit and reveal", variant="primary")
                next_button = gr.Button("Next frame")
            probabilities_label = gr.Label(
                label="The model's picks", num_top_classes=len(CLASSES),
                elem_id="model-picks",
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
    demo.launch(
        theme=gr.themes.Base(
            primary_hue="orange", neutral_hue="slate",
            font=["Segoe UI", "sans-serif"],
        ).set(
            body_background_fill="#faf7f2",
            block_background_fill="#ffffff",
            block_border_color="#e4ddd3",
            input_background_fill="#f5f1ea",
            button_primary_background_fill="#c8703a",
            button_primary_background_fill_hover="#d88752",
            button_primary_text_color="#fffaf5",
            body_background_fill_dark="#111318",
            block_background_fill_dark="#1b1e26",
            block_border_color_dark="#323640",
            input_background_fill_dark="#14171e",
            button_primary_background_fill_dark="#e5ae78",
            button_primary_background_fill_hover_dark="#f2c496",
            button_primary_text_color_dark="#21180f",
            block_radius="16px",
            button_large_radius="12px",
        ),
        css=Path(__file__).with_name("guess_game.css").read_text(),
        footer_links=[],
    )
