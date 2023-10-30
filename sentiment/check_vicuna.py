import re
import json

import torch
import pandas as pd
from fastchat.model import load_model, get_conversation_template, add_model_args


@torch.inference_mode()
def main():
    # Load model
    device = "cuda"
    # root_dir = "/Users/tyler/workspace/oulu"
    root_dir = "/home/tyler/projects/nlp"
    temperature = 0.3
    model, tokenizer = load_model(
        "lmsys/vicuna-7b-v1.5",
        device=device,
        num_gpus=1,
        load_8bit=True,
        cpu_offloading=False,
        debug=False,
    )

    df = pd.read_csv(
        f"{root_dir}/review_test_data.csv",
        dtype=str, na_filter=False)
    reviews = [str(review) for review in df["Review"]]
    results = []

    for review in reviews:
        msg = f"Review: {review}"

        # Build the prompt with a conversation template
        conv = get_conversation_template("lmsys/vicuna-7b-v1.5")
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)

        prompt = open(f"{root_dir}/prompt.txt").read()
        prompt += msg + "\n\n" + "Rating: "

        # Run inference
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1.0,
            max_new_tokens=512,
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        # Print results
        print(f"{conv.roles[0]}: {msg}")
        print(f"{conv.roles[1]}: {outputs}")

        rating = re.findall(r"\d+" + "\n", outputs)
        print(rating)
        if len(rating) > 0:
            try:
                rate = int(re.findall(r"\d+", rating[0])[0])
            except Exception:
                rate = 0
        else:
            rate = 0
        results.append({
            "review": msg,
            "rating": rate,
        })

    json.dump(
        results,
        open(f"{root_dir}/results.json", "w+"),
        indent=4,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
