import datetime
import os
import os.path as osp
import subprocess


def save_git_snapshot(work_dir, job_name, logger):
    """
    save git snapshot to the git repository in work_dir
    use exp/<job_name>_<timestamp> as branch name

    Args:
        work_dir: work directory path
        job_name: job name
        logger: logger
    """
    try:
        git_dir = osp.join(work_dir, ".git")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"exp/{job_name}_{timestamp}"

        project_root = osp.abspath(osp.join(osp.dirname(__file__), "../.."))

        logger.info("=" * 80)
        logger.info(f"save git snapshot to: {work_dir}")
        logger.info(f"project_root: {project_root}")
        logger.info(f"Git branch: {branch_name}")
        logger.info("=" * 80)

        # check if work_dir is already a git repository
        if not osp.exists(git_dir):
            logger.info("[Git] initialize git repository...")
            os.makedirs(work_dir, exist_ok=True)

            # initialize git repository
            subprocess.run(["git", "init"], cwd=work_dir, check=True, capture_output=True)

            # copy code to code_snapshot/Sana
            code_snapshot_dir = osp.join(work_dir, "code_snapshot", "Sana")
            os.makedirs(code_snapshot_dir, exist_ok=True)
            logger.info(f"[Git] copy code to {code_snapshot_dir}...")
            rsync_cmd = [
                "rsync",
                "-av",
                "--exclude=.git",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "--exclude=*.pth",
                "--exclude=*.safetensors",
                "--exclude=output",
                "--exclude=work_dirs",
                "--exclude=*.mp4",
                "--exclude=*.png",
                "--exclude=*.jpg",
                "--exclude=/data",
                f"{project_root}/",
                f"{code_snapshot_dir}/",
            ]
            subprocess.run(rsync_cmd, check=True, capture_output=True)

            # create initial commit and branch
            subprocess.run(["git", "add", "."], cwd=work_dir, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Initial commit - Training started at {timestamp} - Job: {job_name}"],
                cwd=work_dir,
                check=True,
                capture_output=True,
            )
            subprocess.run(["git", "checkout", "-b", branch_name], cwd=work_dir, check=True, capture_output=True)
            subprocess.run(
                ["git", "tag", "-a", f"train_start_{timestamp}", "-m", f"Training started at {timestamp}"],
                cwd=work_dir,
                check=True,
                capture_output=True,
            )

            logger.info(f"[Git] initialize done, create branch: {branch_name}")

        else:
            logger.info("[Git] git repository already exists, update code snapshot...")

            # update code snapshot/Sana
            code_snapshot_dir = osp.join(work_dir, "code_snapshot", "Sana")
            rsync_cmd = [
                "rsync",
                "-av",
                "--exclude=.git",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "--exclude=*.pth",
                "--exclude=*.safetensors",
                "--exclude=output",
                "--exclude=work_dirs",
                "--exclude=*.mp4",
                "--exclude=*.png",
                "--exclude=*.jpg",
                "--exclude=/data",
                f"{project_root}/",
                f"{code_snapshot_dir}/",
            ]
            subprocess.run(rsync_cmd, check=True, capture_output=True)

            # add changes
            subprocess.run(["git", "add", "."], cwd=work_dir, check=True, capture_output=True)

            # check if there are changes
            result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=work_dir, capture_output=True)

            if result.returncode != 0:  # there are changes
                logger.info("[Git] code has changed, create new commit and branch")
                subprocess.run(
                    ["git", "commit", "-m", f"Code snapshot - Training restarted at {timestamp} - Job: {job_name}"],
                    cwd=work_dir,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(["git", "checkout", "-b", branch_name], cwd=work_dir, check=True, capture_output=True)
                subprocess.run(
                    ["git", "tag", "-a", f"train_restart_{timestamp}", "-m", f"Training restarted at {timestamp}"],
                    cwd=work_dir,
                    check=True,
                    capture_output=True,
                )
                logger.info(f"[Git] create new branch: {branch_name}")
            else:
                logger.info("[Git] code has no changes, only create tag record for restart")
                subprocess.run(
                    [
                        "git",
                        "tag",
                        "-a",
                        f"train_restart_{timestamp}",
                        "-m",
                        f"Training restarted at {timestamp} (no code change)",
                    ],
                    cwd=work_dir,
                    check=True,
                    capture_output=True,
                )

        # save git history
        log_output = subprocess.run(
            ["git", "log", "--oneline", "--decorate", "--graph", "--all"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        git_log_file = osp.join(work_dir, f"git_history_{timestamp}.txt")
        with open(git_log_file, "w") as f:
            f.write(f"Git History at {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Branch: {branch_name}\n")
            f.write(f"Job: {job_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(log_output.stdout)

        logger.info(f"[Git] git history saved to: {git_log_file}")
        logger.info(f"[Git] use 'cd {work_dir} && git log --graph --all' to view history")
        logger.info("=" * 80)

    except subprocess.CalledProcessError as e:
        logger.warning(f"[Git] git operation failed: {e}")
        if hasattr(e, "stderr") and e.stderr:
            logger.warning(f"[Git] error output: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")
    except Exception as e:
        logger.warning(f"[Git] error when saving git snapshot: {e}")
