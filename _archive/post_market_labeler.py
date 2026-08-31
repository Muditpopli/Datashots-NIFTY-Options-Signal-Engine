"""
Post-market labeling wrapper.
Run after market close to manually label saved signals for accuracy tracking.
"""

from accuracy_tracker import AccuracyTracker


VALID = {"BULLISH", "BEARISH", "SIDEWAYS", "SKIP", "QUIT"}
VALID_GAP = {"BULLISH", "BEARISH", "NEUTRAL", "SKIP", "QUIT"}


def prompt_direction(signal_id: str, index: str, pred: str, ts: str) -> str:
    print(f"\nSignal: {signal_id}")
    print(f"  Time: {ts}")
    print(f"  Index: {index}")
    print(f"  Predicted: {pred}")
    print("  Actual? [BULLISH/BEARISH/SIDEWAYS] or SKIP or QUIT")
    while True:
        value = input("> ").strip().upper()
        if value in VALID:
            return value
        print("Invalid input. Use BULLISH, BEARISH, SIDEWAYS, SKIP, or QUIT.")


def prompt_gap_direction(gap_id: str, index: str, pred: str, ts: str) -> str:
    print(f"\nGap ID: {gap_id}")
    print(f"  Time: {ts}")
    print(f"  Index: {index}")
    print(f"  Predicted Gap: {pred}")
    print("  Actual next open? [BULLISH/BEARISH/NEUTRAL] or SKIP or QUIT")
    while True:
        value = input("> ").strip().upper()
        if value in VALID_GAP:
            return value
        print("Invalid input. Use BULLISH, BEARISH, NEUTRAL, SKIP, or QUIT.")


def main():
    tracker = AccuracyTracker()
    pending = tracker.get_unlabeled_predictions()
    pending_gap = tracker.get_unlabeled_gap_predictions()

    if not pending and not pending_gap:
        print("No unlabeled predictions found (intraday or gap).")
        return

    print(
        f"Found {len(pending)} unlabeled intraday predictions and "
        f"{len(pending_gap)} unlabeled gap predictions."
    )

    labeled_count = 0
    for row in pending:
        choice = prompt_direction(
            signal_id=row["signal_id"],
            index=row["index"],
            pred=row["pred_direction"],
            ts=row["timestamp"],
        )

        if choice == "QUIT":
            break
        if choice == "SKIP":
            continue

        if tracker.label_outcome(row["signal_id"], choice):
            labeled_count += 1
            print(f"Saved: {row['signal_id']} -> {choice}")

    labeled_gap_count = 0
    for row in pending_gap:
        choice = prompt_gap_direction(
            gap_id=row["gap_id"],
            index=row["index"],
            pred=row["pred_direction"],
            ts=row["timestamp"],
        )

        if choice == "QUIT":
            break
        if choice == "SKIP":
            continue

        if tracker.label_gap_outcome(row["gap_id"], choice):
            labeled_gap_count += 1
            print(f"Saved: {row['gap_id']} -> {choice}")

    stats = tracker.summary()
    gap_stats = tracker.gap_summary()
    print("\nLabeling complete.")
    print(f"Labeled intraday this run: {labeled_count}")
    print(f"Labeled gap this run: {labeled_gap_count}")
    print(f"Intraday total labeled: {stats['labeled_predictions']}")
    print(f"Intraday exact accuracy: {stats['exact_accuracy']:.2%}")
    print(f"Intraday structure accuracy: {stats['structure_accuracy']:.2%}")
    print(f"Intraday directional accuracy: {stats['directional_accuracy']:.2%}")
    print(f"Gap total labeled: {gap_stats['labeled_gap_predictions']}")
    print(f"Gap exact accuracy: {gap_stats['gap_exact_accuracy']:.2%}")


if __name__ == "__main__":
    main()
