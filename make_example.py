"""Generate geometric RGB pixels, solely to test that inference runs."""
import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic.png"))
    args = parser.parse_args()
    image = Image.new("RGB", (224, 224))
    image.putdata([
        (40 + (3*x + y) % 180, 20 + (x + 2*y) % 170, 60 + (x + y*5) % 180)
        for y in range(224) for x in range(224)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")
    print(f"Synthetic test input only: {args.output.resolve()}")


if __name__ == "__main__":
    main()
