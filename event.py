# event.py
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def load_scalars_from_event_file(file_path):
    """Load scalar data from a TensorBoard event file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    event_acc = EventAccumulator(file_path)
    event_acc.Reload()

    scalar_tags = event_acc.Tags().get('scalars', [])
    results = {}

    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        results[tag] = [(event.step, event.value) for event in events]

    return results

if __name__ == "__main__":
    event_file_path = "data/session1/RSM-CoDG/seed3/1/single experiment acc: _test acc/events.out.tfevents.1757340492.autodl-container-f98d4f8817-9b82a446.717187.4"

    try:
        data = load_scalars_from_event_file(event_file_path)
        print("Scalar data:\n")
        for tag, values in data.items():
            print(f"Tag: {tag}")
            for step, value in values:
                print(f"   Subject {step}: {value:.6f}")
    except Exception as e:
        print(f"Error: {e}")