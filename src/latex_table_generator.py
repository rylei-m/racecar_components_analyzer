from ultralytics import YOLO
import argparse

def generate_table(model_path, data_yaml, outfile="table_metrics.tex"):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    names = list(metrics.names.values())
    aps = metrics.box.maps

    with open(outfile, "w") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\begin{tabular}{l c}\n\\hline\n")
        f.write("Class & AP \\\\\n\\hline\n")

        for cls, ap in zip(names, aps):
            f.write(f"{cls} & {ap:.4f} \\\\\n")

        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Per-class Average Precision (AP)}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX table: {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    generate_table(args.model, args.data)


if __name__ == "__main__":
    main()
