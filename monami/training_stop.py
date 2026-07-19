"""Convergence / stop-criteria helpers for DNN training."""

from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional

from monami.config import MLConfig

# Hard ceiling when train-accuracy stop overrides max epochs (prevents runaway fits).
_TARGET_ACCURACY_SAFETY_EPOCHS = 100_000

_manual_stop = threading.Event()


def clear_manual_stop() -> None:
    _manual_stop.clear()


def request_manual_stop() -> None:
    _manual_stop.set()


def is_manual_stop_requested() -> bool:
    return _manual_stop.is_set()


def effective_max_epochs(ml_config: MLConfig) -> int:
    """Epochs passed to ``model.fit`` given the active stop mode."""
    if ml_config.stop_on_train_accuracy:
        return _TARGET_ACCURACY_SAFETY_EPOCHS
    return int(ml_config.epochs)


def stop_mode_summary(ml_config: MLConfig) -> str:
    if ml_config.stop_on_train_accuracy:
        return (
            f"stop on train accuracy ≥ {float(ml_config.target_train_accuracy):.2%} "
            "(overrides max epochs and early stopping)"
        )
    return (
        f"early stopping on val_accuracy (patience={int(ml_config.early_stopping_patience)}), "
        f"max epochs={int(ml_config.epochs)}"
    )


def build_convergence_callbacks(
    ml_config: MLConfig,
    *,
    log_callback: Optional[Callable[[str], None]] = None,
) -> List[Any]:
    """
    Return Keras callbacks for the selected stop criteria, plus manual-stop /
    best-train-accuracy tracking.
    """
    from tensorflow.keras.callbacks import Callback, EarlyStopping

    callbacks: List[Any] = []

    if ml_config.stop_on_train_accuracy:
        target = float(ml_config.target_train_accuracy)

        class _StopOnTrainAccuracy(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                acc = logs.get("accuracy", logs.get("acc"))
                if acc is None:
                    return
                acc_f = float(acc)
                if acc_f >= target:
                    msg = (
                        f"Reached training accuracy {acc_f:.4f} ≥ target {target:.4f} "
                        f"at epoch {epoch + 1}; stopping "
                        "(overrides max epochs and early stopping)."
                    )
                    print(f"[Stop] {msg}", flush=True)
                    if log_callback is not None:
                        log_callback(msg)
                    self.model.stop_training = True

        callbacks.append(_StopOnTrainAccuracy())
    else:
        callbacks.append(
            EarlyStopping(
                monitor="val_accuracy",
                patience=int(ml_config.early_stopping_patience),
                restore_best_weights=True,
            )
        )

    class _BestTrainAccuracyAndManualStop(Callback):
        """Track best training-accuracy weights; honor manual stop requests.

        Registered last so ``on_train_end`` runs after EarlyStopping and can
        override restored weights when the user stops manually.
        """

        def __init__(self):
            super().__init__()
            self.best_acc = -1.0
            self.best_weights = None
            self._manual_stopped = False

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            acc = logs.get("accuracy", logs.get("acc"))
            if acc is not None and float(acc) >= self.best_acc:
                self.best_acc = float(acc)
                self.best_weights = self.model.get_weights()

            if is_manual_stop_requested():
                self._manual_stopped = True
                msg = (
                    f"Manual stop at epoch {epoch + 1} "
                    f"(best train accuracy so far: {self.best_acc:.4f}). "
                    "Restoring those weights."
                )
                print(f"[Stop] {msg}", flush=True)
                if log_callback is not None:
                    log_callback(msg)
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self._manual_stopped and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                msg = f"Restored weights from best training accuracy ({self.best_acc:.4f})."
                print(f"[Stop] {msg}", flush=True)
                if log_callback is not None:
                    log_callback(msg)

    callbacks.append(_BestTrainAccuracyAndManualStop())
    return callbacks
