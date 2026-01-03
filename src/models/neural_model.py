"""
Neural Network Model with MC Dropout

PyTorch MLP with MC Dropout for uncertainty estimation.
Provides diverse predictions compared to tree-based models.
"""

from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False
    logger.warning("PyTorch not available. Install with: pip install torch")

from .base_model import BaseModel


if HAS_TORCH:
    class MCDropoutMLP(nn.Module):
        """
        Multi-layer perceptron with MC Dropout.

        Dropout is applied at inference time for uncertainty estimation.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [128, 64, 32],
            dropout_rate: float = 0.2,
        ):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
                prev_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)

        def predict_with_mc_dropout(
            self,
            x: torch.Tensor,
            n_samples: int = 50,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Make predictions with MC Dropout for uncertainty.

            Args:
                x: Input features
                n_samples: Number of forward passes with dropout

            Returns:
                Tuple of (mean_prediction, std_prediction)
            """
            self.train()  # Keep dropout active

            predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.forward(x)
                    predictions.append(pred)

            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)

            self.eval()  # Reset to eval mode

            return mean_pred, std_pred


class NeuralSpreadModel(BaseModel):
    """
    Neural Network model for NBA spread prediction.

    Uses MC Dropout for uncertainty estimation, providing:
    - Mean prediction
    - Standard deviation of predictions across dropout samples
    """

    DEFAULT_CONFIG = {
        "hidden_dims": [128, 64, 32],
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping_patience": 10,
        "weight_decay": 1e-4,
    }

    def __init__(
        self,
        config: Optional[Dict] = None,
        name: str = "neural_spread",
        mc_samples: int = 50,
        device: Optional[str] = None,
    ):
        """
        Initialize Neural model.

        Args:
            config: Model configuration (overrides defaults)
            name: Model identifier
            mc_samples: Number of MC Dropout samples for uncertainty
            device: Device to use ('cuda', 'mps', or 'cpu')
        """
        super().__init__(name=name)

        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.mc_samples = mc_samples

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional["MCDropoutMLP"] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._training_history: List[Dict] = []

    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features to zero mean and unit variance."""
        if fit:
            self._scaler_mean = np.nanmean(X, axis=0)
            self._scaler_std = np.nanstd(X, axis=0)
            self._scaler_std[self._scaler_std < 1e-6] = 1.0  # Avoid division by zero

        X_scaled = (X - self._scaler_mean) / self._scaler_std
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "NeuralSpreadModel":
        """
        Train the neural network model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features for early stopping
            y_val: Validation labels

        Returns:
            Self for method chaining
        """
        # Validate inputs
        if len(X) == 0:
            raise ValueError("Cannot fit model with empty training data")
        if len(y) == 0:
            raise ValueError("Cannot fit model with empty labels")

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Prepare data
        X_train = self._prepare_features(X).values
        X_train = self._scale_features(X_train, fit=True)
        y_train = y.values.astype(np.float32)

        # Create model
        self.model = MCDropoutMLP(
            input_dim=X_train.shape[1],
            hidden_dims=self.config["hidden_dims"],
            dropout_rate=self.config["dropout_rate"],
        ).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
        )

        # Validation loader if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_prep = self._prepare_features(X_val).values
            X_val_scaled = self._scale_features(X_val_prep)
            y_val_np = y_val.values.astype(np.float32)

            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_scaled),
                torch.FloatTensor(y_val_np),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"])

        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        # Training loop
        self._training_history = []

        for epoch in range(self.config["epochs"]):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_X)

            train_loss /= len(train_dataset)

            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * len(batch_X)

                val_loss /= len(val_dataset)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= self.config["early_stopping_patience"]:
                    logger.info(f"{self.name}: Early stopping at epoch {epoch + 1}")
                    break

            self._training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        self.model.eval()

        # Compute pseudo feature importance via gradient sensitivity
        self._compute_feature_importance(X_train, y_train)

        if self._training_history:
            logger.info(
                f"{self.name}: Trained for {len(self._training_history)} epochs, "
                f"final train_loss={self._training_history[-1]['train_loss']:.4f}"
            )
        else:
            logger.info(f"{self.name}: Training completed with no history recorded")

        return self

    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """
        Compute feature importance via input gradient sensitivity.

        Uses gradient of loss with respect to inputs to estimate importance.
        """
        self.model.eval()

        # Sample subset for efficiency
        n_samples = min(1000, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = torch.FloatTensor(X[indices]).to(self.device)
        y_sample = torch.FloatTensor(y[indices]).to(self.device)

        X_sample.requires_grad = True

        # Forward pass (with gradient tracking for X_sample only)
        outputs = self.model(X_sample)
        loss = nn.BCELoss()(outputs, y_sample)

        # Backward pass
        loss.backward()

        # Feature importance as mean absolute gradient
        importances = X_sample.grad.abs().mean(dim=0).detach().cpu().numpy()

        # Clear gradients to prevent memory leak
        X_sample.grad = None

        self._feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions (mean of MC samples).

        Args:
            X: Features for prediction

        Returns:
            Array of probabilities for positive class
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X).values
        X_scaled = self._scale_features(X_prep)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        mean_pred, _ = self.model.predict_with_mc_dropout(
            X_tensor, n_samples=self.mc_samples
        )

        return mean_pred.cpu().numpy()

    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction uncertainty using MC Dropout.

        Args:
            X: Features for prediction

        Returns:
            Array of uncertainty values (std of MC samples)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X).values
        X_scaled = self._scale_features(X_prep)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        _, std_pred = self.model.predict_with_mc_dropout(
            X_tensor, n_samples=self.mc_samples
        )

        return std_pred.cpu().numpy()

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with uncertainty in one forward pass.

        More efficient than calling predict_proba and get_uncertainty separately.

        Args:
            X: Features for prediction

        Returns:
            Tuple of (probabilities, uncertainties)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X).values
        X_scaled = self._scale_features(X_prep)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        mean_pred, std_pred = self.model.predict_with_mc_dropout(
            X_tensor, n_samples=self.mc_samples
        )

        return mean_pred.cpu().numpy(), std_pred.cpu().numpy()

    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature importance (based on gradient sensitivity)
        """
        if self._feature_importance is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._feature_importance.copy()

    def get_training_history(self) -> pd.DataFrame:
        """Get training history as DataFrame."""
        return pd.DataFrame(self._training_history)

    def _get_save_data(self) -> Dict[str, Any]:
        """Get data for saving."""
        data = super()._get_save_data()
        data.update({
            "config": self.config,
            "mc_samples": self.mc_samples,
            "model_state": self.model.state_dict() if self.model else None,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "training_history": self._training_history,
            "input_dim": len(self.feature_columns) if self.feature_columns else 0,
        })
        return data

    @classmethod
    def _from_save_data(cls, data: Dict[str, Any]) -> "NeuralSpreadModel":
        """Restore model from saved data."""
        instance = cls(
            config=data["config"],
            name=data["name"],
            mc_samples=data.get("mc_samples", 50),
        )
        instance.feature_columns = data["feature_columns"]
        instance.is_fitted = data["is_fitted"]
        instance._feature_importance = data.get("feature_importance")
        instance._scaler_mean = data.get("scaler_mean")
        instance._scaler_std = data.get("scaler_std")
        instance._training_history = data.get("training_history", [])

        # Restore model if state available
        if data.get("model_state") is not None and data.get("input_dim", 0) > 0:
            instance.model = MCDropoutMLP(
                input_dim=data["input_dim"],
                hidden_dims=instance.config["hidden_dims"],
                dropout_rate=instance.config["dropout_rate"],
            ).to(instance.device)
            instance.model.load_state_dict(data["model_state"])
            instance.model.eval()

        return instance

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "config": self.config,
            "mc_samples": self.mc_samples,
            "device": str(self.device),
            "n_epochs_trained": len(self._training_history),
        })
        return config


def create_neural_model(
    config: Optional[Dict] = None,
    name: str = "neural_spread",
    mc_samples: int = 50,
) -> NeuralSpreadModel:
    """
    Factory function to create a Neural Network model.

    Args:
        config: Model configuration
        name: Model name
        mc_samples: Number of MC Dropout samples

    Returns:
        Configured NeuralSpreadModel
    """
    return NeuralSpreadModel(config=config, name=name, mc_samples=mc_samples)
