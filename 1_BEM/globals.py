from pathlib import Path

main_dir = Path.cwd()
# main_dir = Path.cwd().joinpath('1_BEM')
data_dir = main_dir.joinpath("data")
plot_dir = main_dir.joinpath("plots")
res_dir = main_dir.joinpath("results")
verif_dir = main_dir.joinpath("verification")

if __name__ == "__main__":
    print(main_dir)