import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------
# 1. Load Model
# ---------------------------------------------
MODEL_NAME = "SamLowe/roberta-base-go_emotions"

print("Loading Text Emotion Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model Loaded.\n")

# GoEmotions labels
GO_EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

# ---------------------------------------------
# 2. Prediction Function
# ---------------------------------------------
def predict_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]

    # Top 3 emotions
    top_probs, top_ids = torch.topk(probs, k=3)

    results = []
    for i in range(3):
        results.append({
            "emotion": GO_EMOTIONS[top_ids[i]],
            "confidence": float(top_probs[i])
        })

    return results


# ---------------------------------------------
# 3. User Input Loop
# ---------------------------------------------
if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or type 'exit'): ")

        if text.lower() == "exit":
            print("Exiting...")
            break

        results = predict_emotion(text)

        print("\n🔹 Top Emotions:")
        for r in results:
            print(f"➡ {r['emotion']}  ({r['confidence']:.3f})")