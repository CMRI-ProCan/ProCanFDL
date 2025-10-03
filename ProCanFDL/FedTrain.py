"""
Federated Learning Training Module for ProCanFDL
Contains the TrainFedProtNet class for training and evaluating the model
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
from FedModel import FedProtNet


class TrainFedProtNet:
    """
    Training and evaluation class for FedProtNet model
    """

    def __init__(
        self, train_dataset, test_dataset, hypers, load_model=None, save_path=None
    ):
        """
        Initialize the training class

        Args:
            train_dataset: Training dataset (ProtDataset)
            test_dataset: Test dataset (ProtDataset)
            hypers: Dictionary of hyperparameters
            load_model: Path to load pre-trained model (optional)
            save_path: Path to save model (optional)
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.hypers = hypers
        self.save_path = save_path

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = FedProtNet(
            input_dim=train_dataset.feature_dim,
            hidden_dim=hypers["hidden_dim"],
            num_classes=train_dataset.num_classes,
            dropout=hypers["dropout"],
        ).to(self.device)

        # Load pre-trained weights if specified
        if load_model is not None:
            self.load_model(load_model)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=hypers["lr"],
            weight_decay=hypers["weight_decay"],
        )

        # Training parameters
        self.batch_size = hypers["batch_size"]
        self.epochs = hypers["epochs"]

        # Store predictions for confusion matrix
        self.predictions = None

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Calculate AUC (one-vs-rest for multiclass)
        try:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="weighted"
            )
        except:
            auc = 0.0

        # Store predictions for confusion matrix
        self.predictions = all_probs

        return accuracy, f1, auc, all_preds, all_labels, all_probs

    def run_train_val(self):
        """
        Run full training and validation

        Returns:
            dict: Dictionary containing training results
        """
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        best_f1 = 0
        best_epoch = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

            # Save best model
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_epoch = epoch
                if self.save_path is not None:
                    self.save_model()

            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Loss: {train_loss:.4f}, "
                    f"Test Acc: {test_acc:.4f}, "
                    f"Test F1: {test_f1:.4f}, "
                    f"Test AUC: {test_auc:.4f}"
                )

        # Load best model
        if self.save_path is not None:
            self.load_model(self.save_path)

        # Final evaluation
        test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

        return {
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "best_epoch": best_epoch,
        }

    def predict(self):
        """
        Run prediction on test set (used for federated evaluation)

        Returns:
            dict: Dictionary containing test results
        """
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        test_acc, test_f1, test_auc, _, _, _ = self.evaluate(test_loader)

        return {"test_acc": test_acc, "test_f1": test_f1, "test_auc": test_auc}

    def get_pred_for_cm(self):
        """
        Get predictions for confusion matrix

        Returns:
            numpy array: Prediction probabilities
        """
        if self.predictions is None:
            test_loader = DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False
            )
            _, _, _, _, _, _ = self.evaluate(test_loader)

        return self.predictions

    def predict_custom(self, data):
        """
        Make predictions on custom data

        Args:
            data: numpy array or torch tensor of shape (n_samples, n_features)

        Returns:
            tuple: (predicted_classes, probabilities)
        """
        self.model.eval()

        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        data = data.to(self.device)

        with torch.no_grad():
            outputs = self.model(data)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy(), probs.cpu().numpy()

    def save_model(self, path=None):
        """Save model weights"""
        save_path = path if path is not None else self.save_path
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)

    def load_model(self, path):
        """Load model weights"""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

