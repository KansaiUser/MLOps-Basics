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
        
        # Initialize softmax over the correct dimension!
        self.softmax = torch.nn.Softmax(dim=1)
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

        # Forward pass through the model
        # outputs in the entire output object, not just the logits
        with torch.no_grad():  # Disable gradient calculations for inference
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.logits  # Access the logits tensor

        # Apply softmax to get probabilities
        probabilities = self.softmax(logits)  # Shape: [1, num_labels]
        scores = probabilities[0].tolist()    # Convert to list

        # Prepare the predictions
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions

if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    # predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    predictor = ColaPredictor("./models/best-checkpoint.ckpt.ckpt")
    print(predictor.predict(sentence))
