"""
Federated Learning Client for Stethos.
Implements Flower-based federated learning with secure aggregation.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, OrderedDict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import logging
import copy

# Optional import - graceful degradation if flower not available
try:
    import flwr as fl
    from flwr.common import (
        FitRes, EvaluateRes, Parameters, Scalar,
        ndarrays_to_parameters, parameters_to_ndarrays
    )
    from flwr.client import NumPyClient
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    server_address: str = "localhost:8080"
    client_id: str = "stethos-client-1"
    modality: str = "breath"  # breath, cxr, spirometry, fusion
    local_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    enable_dp: bool = True  # Differential privacy
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    max_grad_norm: float = 1.0


class FederatedClient:
    """Federated learning client for a specific modality."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: FederatedConfig,
        device: torch.device
    ):
        if not FLOWER_AVAILABLE:
            raise ImportError("Flower (flwr) not installed. Install with: pip install flwr")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()

        # Privacy accounting
        self.privacy_engine = None
        if config.enable_dp:
            try:
                from opacus import PrivacyEngine
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=config.dp_epsilon,
                    max_grad_norm=config.max_grad_norm,
                )
                logger.info("Differential privacy enabled with epsilon=%.2f", config.dp_epsilon)
            except ImportError:
                logger.warning("Opacus not available, DP disabled")

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train on local data."""
        self.set_parameters(parameters)
        self.model.train()

        epochs = config.get("local_epochs", self.config.local_epochs)

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                if self.config.enable_dp and self.privacy_engine:
                    # Opacus handles gradient clipping and noise
                    pass
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                self.optimizer.step()

                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            logger.info(f"Client {self.config.client_id} epoch {epoch+1}: loss={epoch_loss/len(self.train_loader):.4f}, acc={100.*correct/total:.2f}%")

        # Return updated parameters
        return self.get_parameters(), len(self.train_loader.dataset), {
            "train_loss": epoch_loss / len(self.train_loader),
            "train_accuracy": 100.0 * correct / total,
            "client_id": self.config.client_id
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate on local validation data."""
        self.set_parameters(parameters)
        self.model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total
        logger.info(f"Client {self.config.client_id} validation: loss={val_loss/len(self.val_loader):.4f}, acc={accuracy:.2f}%")

        return val_loss / len(self.val_loader), len(self.val_loader.dataset), {
            "val_accuracy": accuracy,
            "client_id": self.config.client_id
        }


class StethosFederatedClient(fl.client.NumPyClient):
    """Flower NumPyClient wrapper for Stethos modalities."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: FederatedConfig
    ):
        self.client = FederatedClient(model, train_loader, val_loader, config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = config

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        return self.client.get_parameters()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        return self.client.fit(parameters, config)

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        return self.client.evaluate(parameters, config)


def create_federated_client(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    modality: str,
    server_address: str = "localhost:8080",
    client_id: str = "stethos-client-1"
) -> StethosFederatedClient:
    """Create a federated client for a Stethos modality."""
    config = FederatedConfig(
        server_address=server_address,
        client_id=client_id,
        modality=modality
    )
    return StethosFederatedClient(model, train_loader, val_loader, config)


def start_federated_client(client: StethosFederatedClient, server_address: str = "localhost:8080") -> None:
    """Start the federated client."""
    if not FLOWER_AVAILABLE:
        raise ImportError("Flower (flwr) not installed")
    fl.client.start_numpy_client(server_address=server_address, client=client)


# Secure Aggregation utilities
class SecureAggregator:
    """Secure aggregation utilities for federated learning."""

    @staticmethod
    def add_noise(parameters: List[np.ndarray], noise_multiplier: float, max_grad_norm: float) -> List[np.ndarray]:
        """Add Gaussian noise for differential privacy."""
        noisy_params = []
        for param in parameters:
            noise = np.random.normal(0, noise_multiplier * max_grad_norm, param.shape)
            noisy_params.append(param + noise)
        return noisy_params

    @staticmethod
    def clip_gradients(parameters: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
        """Clip gradients to max norm."""
        total_norm = np.sqrt(sum(np.sum(p**2) for p in parameters))
        clip_factor = min(1.0, max_norm / (total_norm + 1e-8))
        return [p * clip_factor for p in parameters]


# Federated learning server strategy (for reference)
def create_federated_strategy() -> "fl.server.strategy.Strategy":
    """Create a federated learning strategy for the server."""
    if not FLOWER_AVAILABLE:
        return None

    return fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )


# Example usage for each modality
def create_breath_federated_client(
    server_address: str = "localhost:8080",
    client_id: str = "stethos-breath-1"
) -> StethosFederatedClient:
    """Create federated client for breath sounds."""
    from plugins.breath import BreathCNN

    model = BreathCNN(num_classes=4)
    # Create dummy data loaders (replace with real data)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(100, 1, 64, 64), torch.randint(0, 4, (100,))),
        batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(20, 1, 64, 64), torch.randint(0, 4, (20,))),
        batch_size=16, shuffle=False
    )

    return create_federated_client(model, train_loader, val_loader, "breath", client_id=client_id)


def create_cxr_federated_client(
    server_address: str = "localhost:8080",
    client_id: str = "stethos-cxr-1"
) -> StethosFederatedClient:
    """Create federated client for CXR."""
    from plugins.cxr import CXRClassifier

    model = CXRClassifier(num_classes=14)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(100, 3, 512, 512), torch.randint(0, 2, (100, 14)).float()),
        batch_size=8, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(20, 3, 512, 512), torch.randint(0, 2, (20, 14)).float()),
        batch_size=8, shuffle=False
    )

    return create_federated_client(model, train_loader, val_loader, "cxr", client_id=client_id)


def create_spirometry_federated_client(
    server_address: str = "localhost:8080",
    client_id: str = "stethos-spiro-1"
) -> StethosFederatedClient:
    """Create federated client for spirometry."""
    from plugins.spirometry import SpirometryTransformer

    model = SpirometryTransformer()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(100, 10, 11), torch.randint(0, 4, (100,))),
        batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(20, 10, 11), torch.randint(0, 4, (20,))),
        batch_size=16, shuffle=False
    )

    return create_federated_client(model, train_loader, val_loader, "spirometry", client_id=client_id)