import os
import argparse


def get_run_name(line):
    line = line.replace("//", "/")
    return line.split("Start testing on ")[1].split("/")[-2]


def get_steps(line):
    EVAL_str = line.split(".pt")[0]
    steps = int(EVAL_str.split("eval_")[1])
    return steps


def parse_result(line):
    res = {}
    line = line.split("|")
    for item in line:
        if item == "" or ":" not in item:
            continue
        key = item.split(":")[0]
        value = item.split(":")[1]
        res[key] = value

    # return f"{res['ssim']},{res['psnr']},{res['fid']},{res['is_mean']},{res['perplexity']},{res['used_codebook']}"
    return (
        f"{res['fid']} & {res['is_mean']} & {res['ssim']} & {res['used_codebook']} \%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./res_token.log")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if "Start testing on" in line:
            print("--------------------")
            print(get_run_name(line))
            continue
        if "Testing on " in line and ".pt" in line:
            steps = get_steps(line)
            continue
        if "!FLAG" in line:
            res = parse_result(line)
            print(f"{steps},{res}")


if __name__ == "__main__":
    main()
