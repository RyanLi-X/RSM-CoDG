# event.py
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def load_scalars_from_event_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no file: {file_path}")

    event_acc = EventAccumulator(file_path)
    event_acc.Reload()

    scalar_tags = event_acc.Tags().get('scalars', [])
    results = {}

    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        results[tag] = [(event.step, event.value) for event in events]

    return results

if __name__ == "__main__":
    event_file_path = ""

    try:
        data = load_scalars_from_event_file(event_file_path)
        print("data：\n")
        for tag, values in data.items():
            print(f"tag: {tag}")
            for step, value in values:
                print(f"   Subject {step}: {value:.6f}")
    except Exception as e:
        print(f"error: {e}")
