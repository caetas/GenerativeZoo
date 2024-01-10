import os
import platform
import pwd
import subprocess  # nosec

from git import Repo


def get_info_repo():  # nosec
    """Get information about git repository."""
    # https://github.com/gitpython-developers/GitPython/issues/633
    repo = Repo(search_parent_directories=True)
    commit_sha = repo.head.object.hexsha
    # https://gist.github.com/pwithnall/7bc5f320b3bdf418265a
    version = str(sorted(repo.tags, key=lambda t: t.commit.committed_datetime)[-1])
    try:
        branch_name = repo.active_branch.name
    except Exception:
        branch_name = "DETACHED_" + repo.head.object.hexsha
    repo_url = repo.remotes[0].config_reader.get("url")
    project_name = repo.remotes.origin.url.split(".git")[0].split("/")[-1]
    try:
        git_user = subprocess.check_output(["git", "config", "user.name"]).decode("utf-8").replace("\n", "")  # nosec
    except Exception:
        git_user = "jenkins_ci"

    base_url = (
        repo_url.split(".git")[0]
        .replace("ssh://git@git.", "https://bitbucket.")
        .replace(".fraunhofer.pt/", ".fraunhofer.pt/projects/")
    )

    commit_diff_url = base_url.replace(project_name, f"repos/{project_name}/commits/{commit_sha}")
    code_snapshot_url = base_url.replace(project_name, f"repos/{project_name}/browse?at={commit_sha}")

    git_info = {
        "commit_sha": commit_sha,
        "version": version,
        "branch_name": branch_name,
        "repo_url": repo_url,
        "project_name": project_name,
        "git_user": git_user,
        "commit_diff_url": commit_diff_url,
        "code_snapshot_url": code_snapshot_url,
    }

    return git_info


def get_info_machine():
    """Get information about the machine."""
    username = pwd.getpwuid(os.getuid()).pw_name
    uname = platform.uname()
    op_system = uname.system
    node = uname.node
    kernel = uname.release

    return username, op_system, node, kernel


def get_info_software():
    """Get information about software stack."""
    python_version = platform.python_version()

    return python_version


# EOF
