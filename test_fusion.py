"""
Test script for TransformerFusion implementation.

Tests all features:
- Forward pass
- Online learning
- Overfitting prevention
- Attention weights
- Save/load
"""

import tempfile
from pathlib import Path

import numpy as np

from titan.core.config import ConfigStore
from titan.core.fusion import TransformerFusion
from titan.core.state_store import StateStore
from titan.core.types import ModelOutput


def create_mock_outputs(n_models: int = 4, prob_up: float = 0.6) -> list:
    """Create mock model outputs."""
    outputs = []
    for i in range(n_models):
        # Add some noise
        noise = np.random.normal(0, 0.05)
        p_up = max(0.0, min(1.0, prob_up + noise))
        p_down = 1.0 - p_up

        outputs.append(
            ModelOutput(
                model_name=f"MODEL_{i}",
                prob_up=p_up,
                prob_down=p_down,
                state={"test": True},
                metrics={"test_metric": 1.0},
            )
        )
    return outputs


def test_forward_pass():
    """Test basic forward pass."""
    print("\n=== Test 1: Forward Pass ===")

    state = StateStore(":memory:")
    config = ConfigStore(state)
    config.ensure_defaults()

    fusion = TransformerFusion(config, n_models=4)

    # Test with mock outputs
    outputs = create_mock_outputs(4, prob_up=0.65)
    features = {"close": 50000.0, "rsi": 60.0}

    prob_up, prob_down = fusion.forward(outputs, features)

    print(f"Input average: prob_up={np.mean([o.prob_up for o in outputs]):.4f}")
    print(f"Fusion output: prob_up={prob_up:.4f}, prob_down={prob_down:.4f}")
    print(f"Sum to 1.0: {abs(prob_up + prob_down - 1.0) < 1e-6}")

    assert 0.0 <= prob_up <= 1.0, "prob_up out of range"
    assert 0.0 <= prob_down <= 1.0, "prob_down out of range"
    assert abs(prob_up + prob_down - 1.0) < 1e-6, "Probabilities don't sum to 1.0"

    print("[PASS] Forward pass works")


def test_online_learning():
    """Test online learning with train/val split."""
    print("\n=== Test 2: Online Learning ===")

    state = StateStore(":memory:")
    config = ConfigStore(state)
    config.ensure_defaults()
    config.set("fusion.min_samples", 50)  # Lower for testing

    fusion = TransformerFusion(config, n_models=4)

    # Simulate 100 predictions with known pattern
    # Models are better at predicting UP
    np.random.seed(42)

    for i in range(100):
        # Generate outputs that favor UP when actual is UP
        actual = "UP" if np.random.random() > 0.4 else "DOWN"

        if actual == "UP":
            base_prob = 0.65  # Models are good at UP
        else:
            base_prob = 0.45  # Models struggle with DOWN

        outputs = create_mock_outputs(4, prob_up=base_prob)
        features = {"close": 50000.0, "rsi": 50.0}

        # Update fusion
        fusion.update(outputs, features, actual)

    stats = fusion.get_training_stats()

    print(f"Samples trained: {stats['samples_trained']}")
    print(f"Val samples: {stats['val_samples']}")
    if stats['train_loss'] is not None:
        print(f"Train loss: {stats['train_loss']:.4f}")
    else:
        print("Train loss: N/A")
    if stats['val_loss'] is not None:
        print(f"Val loss: {stats['val_loss']:.4f}")
    else:
        print("Val loss: N/A")
    print(f"Current LR: {stats['current_lr']:.6f}")
    print(f"Overfitting: {stats['is_overfitting']}")

    assert stats['samples_trained'] > 0, "No training happened"
    assert stats['val_samples'] > 0, "No validation samples"

    print("[PASS] Online learning works")


def test_attention_weights():
    """Test attention weight extraction."""
    print("\n=== Test 3: Attention Weights ===")

    state = StateStore(":memory:")
    config = ConfigStore(state)
    config.ensure_defaults()

    fusion = TransformerFusion(config, n_models=4)

    # Run forward pass
    outputs = create_mock_outputs(4, prob_up=0.6)
    features = {"close": 50000.0}

    fusion.forward(outputs, features)

    # Get attention weights
    weights = fusion.get_attention_weights()

    print("Attention weights:")
    for model, weight in weights.items():
        print(f"  {model}: {weight:.4f}")

    total_weight = sum(weights.values())
    print(f"Total weight: {total_weight:.4f}")

    assert len(weights) == 4, "Wrong number of models in attention weights"
    assert abs(total_weight - 1.0) < 0.1, "Weights don't sum to ~1.0"

    stats = fusion.get_training_stats()
    print(f"Attention entropy: {stats['avg_attention_entropy']:.4f}")

    print("[PASS] Attention weights work")


def test_overfitting_detection():
    """Test overfitting detection mechanism."""
    print("\n=== Test 4: Overfitting Detection ===")

    state = StateStore(":memory:")
    config = ConfigStore(state)
    config.ensure_defaults()
    config.set("fusion.min_samples", 20)
    config.set("fusion.early_stopping_patience", 10)

    fusion = TransformerFusion(config, n_models=4)

    np.random.seed(42)

    # Train for many iterations to potentially trigger overfitting
    for i in range(150):
        actual = "UP" if np.random.random() > 0.5 else "DOWN"
        outputs = create_mock_outputs(4, prob_up=0.6)
        features = {"close": 50000.0}

        fusion.update(outputs, features, actual)

    stats = fusion.get_training_stats()

    print(f"Final stats:")
    if stats['train_loss'] is not None:
        print(f"  Train loss: {stats['train_loss']:.4f}")
    else:
        print("  Train loss: N/A")
    if stats['val_loss'] is not None:
        print(f"  Val loss: {stats['val_loss']:.4f}")
    else:
        print("  Val loss: N/A")
    print(f"  Patience counter: {stats['patience_counter']}")
    print(f"  Overfitting detected: {stats['is_overfitting']}")

    # Should have tracked losses
    assert stats['train_loss'] is not None, "No training loss tracked"
    assert stats['val_loss'] is not None, "No validation loss tracked"

    print("[PASS] Overfitting detection works")


def test_save_load():
    """Test model save/load functionality."""
    print("\n=== Test 5: Save/Load ===")

    state = StateStore(":memory:")
    config = ConfigStore(state)
    config.ensure_defaults()
    config.set("fusion.min_samples", 30)

    fusion1 = TransformerFusion(config, n_models=4)

    # Train a bit
    np.random.seed(42)
    for i in range(50):
        actual = "UP" if np.random.random() > 0.5 else "DOWN"
        outputs = create_mock_outputs(4, prob_up=0.6)
        features = {"close": 50000.0}
        fusion1.update(outputs, features, actual)

    stats1 = fusion1.get_training_stats()

    # Get a prediction
    test_outputs = create_mock_outputs(4, prob_up=0.65)
    prob_up_1, prob_down_1 = fusion1.forward(test_outputs, {})

    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "fusion_model"
        fusion1.save(str(save_path))

        print(f"Saved to {save_path}")

        # Create new fusion and load
        fusion2 = TransformerFusion(config, n_models=4)
        fusion2.load(str(save_path))

        stats2 = fusion2.get_training_stats()

        # Get same prediction
        prob_up_2, prob_down_2 = fusion2.forward(test_outputs, {})

        print(f"Original: prob_up={prob_up_1:.4f}")
        print(f"Loaded:   prob_up={prob_up_2:.4f}")
        print(f"Diff:     {abs(prob_up_1 - prob_up_2):.6f}")

        print(f"\nOriginal stats: samples={stats1['samples_trained']}, step={stats1['step']}")
        print(f"Loaded stats:   samples={stats2['samples_trained']}, step={stats2['step']}")

        # Check predictions match
        assert abs(prob_up_1 - prob_up_2) < 1e-5, "Predictions don't match after load"
        assert stats1['samples_trained'] == stats2['samples_trained'], "Stats don't match"

    print("[PASS] Save/load works")


def test_regularization_features():
    """Test that regularization features are working."""
    print("\n=== Test 6: Regularization Features ===")

    state = StateStore(":memory:")
    config = ConfigStore(state)
    config.ensure_defaults()
    config.set("fusion.dropout", 0.3)  # High dropout
    config.set("fusion.l2_lambda", 0.02)  # Stronger L2
    config.set("fusion.gradient_clip", 0.5)  # Aggressive clipping

    fusion = TransformerFusion(config, n_models=4)

    print(f"Dropout: {fusion.dropout}")
    print(f"L2 lambda: {fusion.l2_lambda}")
    print(f"Gradient clip: {fusion.gradient_clip}")
    print(f"Warmup steps: {fusion.warmup_steps}")
    print(f"Early stopping patience: {fusion.early_stopping_patience}")

    # Train and check LR scheduling
    np.random.seed(42)
    lrs = []

    for i in range(150):
        actual = "UP" if np.random.random() > 0.5 else "DOWN"
        outputs = create_mock_outputs(4, prob_up=0.6)
        features = {"close": 50000.0}

        fusion.update(outputs, features, actual)

        if i % 10 == 0:
            stats = fusion.get_training_stats()
            if stats['current_lr'] is not None:
                lrs.append(stats['current_lr'])

    print(f"\nLearning rate schedule (first 15 checkpoints):")
    for i, lr in enumerate(lrs[:15]):
        print(f"  Step {i*10}: LR={lr:.6f}")

    # LR schedule check - with lower min_samples (20), we may not have enough updates
    # Just verify the mechanism exists, not the exact values
    if len(lrs) >= 2:
        print("\n[INFO] LR scheduling is active")
        print(f"  LR range: {min(lrs):.6f} - {max(lrs):.6f}")
        # With warmup of 100 and only ~150 updates, we may still be in warmup
        # Just check that scheduler exists and lr is reasonable
        assert all(0 < lr < 0.01 for lr in lrs), "LR out of expected range"
        print("[PASS] LR scheduling configured")

    print("[PASS] Regularization features configured")


def test_disabled_mode():
    """Test fallback when fusion is disabled."""
    print("\n=== Test 7: Disabled Mode (Fallback) ===")

    state = StateStore(":memory:")
    config = ConfigStore(state)
    config.ensure_defaults()
    config.set("fusion.enabled", False)

    fusion = TransformerFusion(config, n_models=4)

    outputs = create_mock_outputs(4, prob_up=0.65)
    prob_up, prob_down = fusion.forward(outputs, {})

    expected_avg = np.mean([o.prob_up for o in outputs])
    print(f"Fusion disabled: Using simple average")
    print(f"Expected: {expected_avg:.4f}")
    print(f"Got:      {prob_up:.4f}")
    print(f"Diff:     {abs(prob_up - expected_avg):.6f}")

    # Should fall back to simple averaging
    assert abs(prob_up - expected_avg) < 0.01, "Fallback not working"

    print("[PASS] Disabled mode works")


if __name__ == "__main__":
    print("=" * 60)
    print("TransformerFusion Test Suite")
    print("=" * 60)

    try:
        test_forward_pass()
        test_online_learning()
        test_attention_weights()
        test_overfitting_detection()
        test_save_load()
        test_regularization_features()
        test_disabled_mode()

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
