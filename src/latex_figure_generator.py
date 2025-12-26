from pathlib import Path
import argparse

def generate_figure_block(image_dir="visualizations", outfile="figures.tex"):
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob("*.jpg"))

    with open(outfile, "w") as f:
        for img in images:
            f.write("\\begin{figure}[H]\n")
            f.write("\\centering\n")
            f.write(f"\\includegraphics[width=0.85\\textwidth]{{{img}}}\n")
            f.write(f"\\caption{{YOLOv8 predictions on {img.name}}}\n")
            f.write("\\end{figure}\n\n")

    print(f"Generated LaTeX figure block in {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="visualizations")
    parser.add_argument("--out", default="figures.tex")
    args = parser.parse_args()

    generate_figure_block(args.dir, args.out)


if __name__ == "__main__":
    main()
