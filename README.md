# Agentic AI Bootcamp

This bootcamp is a hands-on path from inference to research agents. You will call **NVIDIA® NIM™** from local endpoints, expose and consume capabilities with the **Model Context Protocol (MCP)** (including a low-level server implementation), and orchestrate reasoning and tool use with **OpenCode**. You will then use **agentic skills**.

## Prerequisites

Participants are expected to have Python programming knowledge, basic prior knowledge of SQL databases and queries, Natural Language Processing knowledge, and NVIDIA NGC and API keys.

Ensure the following tools are installed on your system:
* [Python](https://docs.astral.sh/uv/guides/install-python/)
* [Git Version Control](https://github.com/git-guides/install-git)
* Node Package eXecute (npx)
* [Docker](https://docs.docker.com/engine/install/)
* [Docker rootless mode](https://docs.docker.com/engine/security/rootless/)
* [NVIDIA Container Toolkit - with rootless mode](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* [JupyterLab](https://jupyter.org/install) (see below for pip install)


## Labs

### Install Brev cli on your local machine (only needed if using Brev)

1. Open [brev.dev](https://brev.nvidia.com) and create / log into your account.
2. Follow the steps in the **CLI Getting started** guide [here](https://docs.nvidia.com/brev/cli/getting-started) to install the Brev CLI which will allow you manage your instances, including the port forwarding rules.

### Clone the git repository

Open a terminal and:

```bash
git clone https://github.com/jbar-nv/agents-for-research-bootcamp
cd agentic-ai-bootcamp
```

### Setup environment variables (not needed in Brev)

For a multi-tenant setup, the below ports will need to be dynamically allocated.

```
BOOTCAMP_DIR=$PWD # path to agentic-ai-bootcamp
JUPYTER_PORT=8888 # Port to access jupyter lab
export MCP_PORT=8001 # Port to access HTTP based MCP Server
export NIM_PORT=8000 # Port to access NIM server
# export INF_URL="http://${HOSTNAME}:${NIM_PORT}/v1" # Local NIM server URL
export INF_URL="https://integrate.api.nvidia.com/v1" # if using NVIDIA cloud endpoint
export MODEL_ID=nvidia/nemotron-3.5-lightning
```

### Create Python virtual environment (not needed on brev instance)
```bash
python3 -m venv agentic-ai-venv
source ./agentic-ai-venv/bin/activate
```

### Install Python modules (not needed on brev instance)

```bash
sudo apt install python3-pip
pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Install npx

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# you might need to exit and re-open a terminal for that
nvm install --lts
```

### Install OpenCode 2 (coding harness)

```bash
curl -fsSL https://opencode.ai/v2/install | bash
```

### Start Jupyter labs (not needed on brev instance)

```bash
jupyter lab --ip 0.0.0.0 --port $JUPYTER_PORT --no-browser --ServerApp.token='' --ServerApp.password=''
```

### Opening the labs

With Jupyter lab running, open http://$HOSTNAME:$JUPYTER_PORT. In the workspace, open the `tutorial` directory and start from `start_here.ipynb`.

When you are finished with the labs close your shell or pressing Ctrl+D in the terminal. 

Congratulations, you've successfully built and deployed an Agentic AI Bootcamp!

### Source Material

Some material are inspired from
- https://github.com/openhackathons-org/agentic-ai-bootcamp/
- https://github.com/dlab-berkeley/Agentic-AI-Research-Workflows/
