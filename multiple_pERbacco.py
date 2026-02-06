from argparse import ArgumentParser
import subprocess


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["all", "cora", "camera", "funding", "voters","wdc80", "synth_10000"])

    args = parser.parse_args()
    dataset = args.dataset

if dataset == "all":
    datasets = ["cora", "camera", "funding", "voters","wdc80", "synth_10000"]
else:
    datasets = [dataset]

# Loop over each dataset and run the script
for dataset in datasets:
    #for batch_size in ["40", "20", "5"]:
    for batch_size in ["10"]:
            
        # ONLINE/PERBAC
        if True: 
            for mu_benefit in ["brmean","brmax"]:
                if "synth" in dataset:
                    list_synth_precision = ["1","0.5","0.2","0.05"]
                else:
                    list_synth_precision = ["False"]
                for synth_precision in list_synth_precision:
                    subprocess.run(["python", "perbacco.py", "--dataset", dataset, "--batch_size", batch_size, 
                                    "--alg_community", "False", "--lambda_w", "False", "--mu_benefit", mu_benefit,  "--optimal", "False",
                                    "--synth_precision", synth_precision])
        
            
            # PERBACCO
            if True: 
                for alg_community in ["louvain"]:                    
                    for lambda_w in ["0.05"]:
                        if "synth" in dataset:
                            list_synth_precision = ["1","0.5","0.2","0.05"]
                        else:
                            list_synth_precision = ["False"]
                        for synth_precision in list_synth_precision:
                            for mu_benefit in ["brmean"]:
                                subprocess.run(["python", "perbacco.py", "--dataset", dataset, "--batch_size", batch_size, 
                                                "--alg_community", alg_community, "--lambda_w", lambda_w, "--mu_benefit", mu_benefit, "--optimal", "False",
                                                "--synth_precision", synth_precision])
        
        # OPTIMAL SOLUTION
        if True:
            if "synth" in dataset:
                list_synth_precision = ["1"] # DO NOT CHANGE!!!!!!
            else:
                list_synth_precision = ["False"]
            for synth_precision in list_synth_precision:
                subprocess.run(["python", "perbacco.py", "--dataset", dataset, "--batch_size", batch_size, 
                                "--alg_community", "False", "--lambda_w", "False", "--mu_benefit", "brmax", "--optimal", "True",
                                "--synth_precision", synth_precision])



