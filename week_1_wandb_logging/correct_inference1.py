import torch
from model import ColaModel
from data import DataModule

class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        # Ensures sofmax is appied across the class logits for each example
        self.softmax = torch.nn.Softmax(dim=1)  # Softmax for batch of predictions
        self.labels = ["unacceptable", "acceptable"]

        # Get the device the model is on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to the appropriate device

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        # Move input tensors to the same device as the model
        input_ids = torch.tensor([processed["input_ids"]]).to(self.device)
        attention_mask = torch.tensor([processed["attention_mask"]]).to(self.device)

        # Pass tensors to the model
        # The logits are directly accesible
        logits = self.model(input_ids, attention_mask).logits  # Get logits directly
        scores = self.softmax(logits).tolist()[0]  # Apply softmax to logits for each class

        # Prepare predictions with labels
        predictions = [{"label": label, "score": score} for label, score in zip(self.labels, scores)]
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    # predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    predictor = ColaPredictor("./models/best-checkpoint.ckpt.ckpt")
    print(predictor.predict(sentence))
