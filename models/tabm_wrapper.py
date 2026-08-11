"""TabM model wrapper with training and evaluation utilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple, Optional, Any
import numpy as np
from tabm import TabM
from rtdl_num_embeddings import LinearReLUEmbeddings


class TabMWrapper:
    """Wrapper around TabM for training and inference."""
    
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Optional[list] = None,
        d_out: int = 1,
        task_type: str = "binclass",
        k: int = 32,
        arch_type: str = "tabm",
        n_blocks: int = 3,
        d_block: int = 512,
        dropout: float = 0.1,
        lr: float = 2e-3,
        weight_decay: float = 3e-4,
        device: str = "cpu",
        use_embeddings: bool = False,
    ):
        """Initialize TabM wrapper.
        
        Args:
            n_num_features: Number of numerical features.
            cat_cardinalities: List of cardinalities for categorical features.
            d_out: Output dimension (1 for binary/regression, n_classes for multiclass).
            task_type: 'binclass', 'multiclass', or 'regression'.
            k: Number of ensemble members.
            arch_type: 'tabm', 'tabm-mini', or 'tabm-packed'.
            n_blocks: Number of MLP blocks.
            d_block: Hidden dimension.
            dropout: Dropout rate.
            lr: Learning rate.
            weight_decay: Weight decay.
            device: Device for training.
            use_embeddings: If True, use LinearReLUEmbeddings (TabM†). If False, raw features (TabM).
        """
        self.task_type = task_type
        self.k = k
        self.device = device
        self.d_out = d_out
        
        # Build model
        if use_embeddings and n_num_features > 0:
            num_embeddings = LinearReLUEmbeddings(n_num_features)
        else:
            num_embeddings = None
            
        self.model = TabM.make(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            num_embeddings=num_embeddings,
            d_out=d_out,
            k=k,
            arch_type=arch_type,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.history = {"train_loss": [], "val_loss": [], "val_metric": []}
    
    def _prepare_batch(
        self,
        X_num: np.ndarray,
        X_cat: Optional[np.ndarray],
        Y: np.ndarray,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Convert numpy arrays to torch tensors."""
        x_num = torch.from_numpy(X_num).to(self.device)
        if X_cat is not None:
            x_cat = torch.from_numpy(X_cat).long().to(self.device)
        else:
            x_cat = None
        y = torch.from_numpy(Y).to(self.device)
        return x_num, x_cat, y
    
    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss per ensemble member (mean over k).
        
        This is the CRITICAL TabM training detail: optimize the mean loss,
        NOT the loss of the mean prediction.
        
        y_pred: (B, k, d_out)
        y_true: (B,) or (B, 1)
        """
        batch_size, k, d_out = y_pred.shape
        
        if self.task_type == "regression":
            # y_true should be (B, 1) for broadcasting
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            # Expand to (B, k, 1)
            y_true_expanded = y_true.unsqueeze(1).expand(batch_size, k, d_out)
            loss = F.mse_loss(y_pred, y_true_expanded, reduction="none").mean(dim=(1, 2)).mean()
        elif self.task_type == "binclass":
            # Binary classification with single output
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            y_true_expanded = y_true.unsqueeze(1).expand(batch_size, k, d_out)
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true_expanded.float(), reduction="none").mean(dim=(1, 2)).mean()
        elif self.task_type == "multiclass":
            # Multi-class: y_pred is (B, k, n_classes)
            # Reshape to (B*k, n_classes) and repeat labels
            y_pred_2d = y_pred.reshape(-1, d_out)  # (B*k, n_classes)
            y_true_repeated = y_true.unsqueeze(1).expand(batch_size, k).reshape(-1)  # (B*k,)
            loss = F.cross_entropy(y_pred_2d, y_true_repeated, reduction="mean")
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        
        return loss
    
    def fit(
        self,
        X_num_train: np.ndarray,
        X_cat_train: Optional[np.ndarray],
        Y_train: np.ndarray,
        X_num_val: np.ndarray,
        X_cat_val: Optional[np.ndarray],
        Y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 256,
        patience: int = 20,
        eval_every: int = 1,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            X_num_train, X_cat_train, Y_train: Training data.
            X_num_val, X_cat_val, Y_val: Validation data.
            epochs: Maximum number of epochs.
            batch_size: Batch size.
            patience: Early stopping patience.
            eval_every: Evaluate validation every N epochs (1 = every epoch).
            verbose: Print progress.
            
        Returns:
            Training history dict.
        """
        assert eval_every >= 1, "eval_every must be >= 1"
        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_num_train),
            torch.from_numpy(Y_train) if Y_train is not None else torch.empty(0),
        )
        if X_cat_train is not None:
            train_dataset = TensorDataset(
                torch.from_numpy(X_num_train),
                torch.from_numpy(X_cat_train).long(),
                torch.from_numpy(Y_train),
            )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                if X_cat_train is not None:
                    x_num, x_cat, y = batch
                    x_num = x_num.to(self.device)
                    x_cat = x_cat.to(self.device)
                    y = y.to(self.device)
                else:
                    x_num, y = batch
                    x_num = x_num.to(self.device)
                    x_cat = None
                    y = y.to(self.device)
                
                self.optimizer.zero_grad()
                y_pred = self.model(x_num, x_cat)  # (B, k, d_out)
                loss = self._compute_loss(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            
            # Validation (only every eval_every epochs for speed)
            do_eval = (epoch % eval_every == 0) or (epoch == epochs - 1)
            
            if do_eval:
                val_loss, val_metric = self.evaluate(X_num_val, X_cat_val, Y_val, batch_size=batch_size)
            else:
                val_loss = self.history["val_loss"][-1] if self.history["val_loss"] else float('inf')
                val_metric = self.history["val_metric"][-1] if self.history["val_metric"] else 0.0
            
            self.history["train_loss"].append(np.mean(train_losses))
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)
            
            if do_eval and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            if verbose and (epoch % 10 == 0 or epoch < 5 or do_eval):
                eval_marker = " *" if do_eval else "  "
                print(f"Epoch {epoch:3d}{eval_marker}| Train Loss: {np.mean(train_losses):.4f} | Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}")
            
            # Early stopping: if no improvement for 'patience' epochs
            # Note: with eval_every > 1, patience is still in raw epochs
            if epoch - best_epoch >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self.history
    
    def evaluate(
        self,
        X_num: np.ndarray,
        X_cat: Optional[np.ndarray],
        Y: np.ndarray,
        batch_size: int = 256,
    ) -> Tuple[float, float]:
        """Evaluate on a dataset.
        
        Returns:
            (loss, metric) where metric is accuracy for classification or negative RMSE for regression.
        """
        self.model.eval()
        
        dataset = TensorDataset(torch.from_numpy(X_num), torch.from_numpy(Y))
        if X_cat is not None:
            dataset = TensorDataset(
                torch.from_numpy(X_num),
                torch.from_numpy(X_cat).long(),
                torch.from_numpy(Y),
            )
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                if X_cat is not None:
                    x_num, x_cat, y = batch
                    x_num = x_num.to(self.device)
                    x_cat = x_cat.to(self.device)
                    y = y.to(self.device)
                else:
                    x_num, y = batch
                    x_num = x_num.to(self.device)
                    x_cat = None
                    y = y.to(self.device)
                
                y_pred = self.model(x_num, x_cat)  # (B, k, d_out)
                loss = self._compute_loss(y_pred, y)
                total_loss += loss.item() * len(y)
                n_samples += len(y)
                
                # For metrics, average predictions across ensemble
                if self.task_type == "regression":
                    pred = y_pred.mean(dim=1).squeeze(-1)  # (B,)
                elif self.task_type == "binclass":
                    pred = torch.sigmoid(y_pred).mean(dim=1).squeeze(-1)  # (B,)
                elif self.task_type == "multiclass":
                    pred = F.softmax(y_pred, dim=2).mean(dim=1)  # (B, n_classes)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        avg_loss = total_loss / n_samples
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Compute metric
        if self.task_type == "regression":
            metric = -np.sqrt(np.mean((all_preds - all_targets) ** 2))  # Negative RMSE (higher is better)
        elif self.task_type == "binclass":
            metric = np.mean((all_preds > 0.5).astype(int) == all_targets.astype(int))
        elif self.task_type == "multiclass":
            metric = np.mean(all_preds.argmax(axis=1) == all_targets)
        
        return avg_loss, metric
    
    def predict(
        self,
        X_num: np.ndarray,
        X_cat: Optional[np.ndarray],
        batch_size: int = 256,
        return_ensemble: bool = False,
    ) -> np.ndarray:
        """Make predictions.
        
        Args:
            X_num, X_cat: Input features.
            batch_size: Inference batch size.
            return_ensemble: If True, return (B, k, d_out) ensemble predictions.
                             If False, return averaged predictions.
                             
        Returns:
            Predictions. For classification with return_ensemble=False, returns probabilities.
        """
        self.model.eval()
        
        if X_cat is not None:
            dataset = TensorDataset(
                torch.from_numpy(X_num),
                torch.from_numpy(X_cat).long(),
            )
        else:
            dataset = TensorDataset(torch.from_numpy(X_num))
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []
        
        with torch.no_grad():
            for batch in loader:
                if X_cat is not None:
                    x_num, x_cat = batch
                    x_num = x_num.to(self.device)
                    x_cat = x_cat.to(self.device)
                else:
                    x_num = batch[0].to(self.device)
                    x_cat = None
                
                y_pred = self.model(x_num, x_cat)  # (B, k, d_out)
                
                if not return_ensemble:
                    # Average across ensemble
                    if self.task_type == "regression":
                        y_pred = y_pred.mean(dim=1)  # (B, d_out)
                    elif self.task_type == "binclass":
                        y_pred = torch.sigmoid(y_pred).mean(dim=1)  # (B, d_out)
                    elif self.task_type == "multiclass":
                        y_pred = F.softmax(y_pred, dim=2).mean(dim=1)  # (B, n_classes)
                
                all_preds.append(y_pred.cpu().numpy())
        
        return np.concatenate(all_preds)
    
    def predict_ensemble(
        self,
        X_num: np.ndarray,
        X_cat: Optional[np.ndarray],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return raw ensemble predictions for UQ analysis.
        
        Returns:
            Array of shape (B, k, d_out) with logits for classification or raw values for regression.
        """
        return self.predict(X_num, X_cat, batch_size, return_ensemble=True)
    
    def save(self, path: str) -> None:
        """Save model state."""
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "config": {
                "task_type": self.task_type,
                "k": self.k,
                "d_out": self.d_out,
            },
        }, path)
    
    def load(self, path: str) -> None:
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint["history"]
