import re


def tracker(args, result_dict, label="", pattern="epoch_step", metric="FID"):
    if args.report_to == "wandb":
        import wandb

        wandb_name = f"[{args.log_metric}]_{args.name}"
        wandb.init(project=args.tracker_project_name, name=wandb_name, resume="allow", id=wandb_name, tags="metrics")
        run = wandb.run
        if pattern == "step":
            pattern = "sample_steps"
        elif pattern == "epoch_step":
            pattern = "step"
        custom_name = f"custom_{pattern}"
        run.define_metric(custom_name)
        # define which metrics will be plotted against it
        run.define_metric(f"{metric}_{label}", step_metric=custom_name)

        steps = []
        results = []

        def extract_value(regex, exp_name):
            match = re.search(regex, exp_name)
            if match:
                return match.group(1)
            else:
                return "unknown"

        for exp_name, result_value in result_dict.items():
            if pattern == "step":
                regex = r".*step(\d+)_scale.*"
                custom_x = extract_value(regex, exp_name)
            elif pattern == "sample_steps":
                regex = r".*step(\d+)_size.*"
                custom_x = extract_value(regex, exp_name)
            else:
                regex = rf"{pattern}(\d+(\.\d+)?)"
                custom_x = extract_value(regex, exp_name)
                custom_x = 1 if custom_x == "unknown" else custom_x

            assert custom_x != "unknown"
            steps.append(float(custom_x))
            results.append(result_value)

        sorted_data = sorted(zip(steps, results))
        steps, results = zip(*sorted_data)

        for step, result in sorted(zip(steps, results)):
            run.log({f"{metric}_{label}": result, custom_name: step})
    else:
        print(f"{args.report_to} is not supported")
