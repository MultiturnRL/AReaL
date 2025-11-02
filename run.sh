#!/bin/bash

set -euox pipefail

export TRAIN_SCRIPT=experiments/train.py
export CONFIG=experiments/config.yaml
export EXPERIMENT_NAME=grpo_multiturn
export TRIAL_NAME=test1
export SANDBOX_CONTROL_PLANE=http://a8b4ee4606659412ca97fd13254655e1-1332289600.us-west-2.elb.amazonaws.com
export SANDBOX_GATEWAY=http://a8ba3420c90c64b2da804891c7d96d2f-1217634634.us-west-2.elb.amazonaws.com
export SANDBOX_FUSION_URL=https://sandbox.broyojo.com/run_code

uv run --no-sync python3 -c "from experiments.sandbox import Sandbox; import asyncio; import os; asyncio.run(Sandbox(os.environ['SANDBOX_CONTROL_PLANE']).deprovision_all())"

uv run --no-sync python3 -m areal.launcher.local $TRAIN_SCRIPT --config $CONFIG \
    experiment_name=$EXPERIMENT_NAME \
    trial_name=$TRIAL_NAME \
    sandbox_control_plane=$SANDBOX_CONTROL_PLANE \
    sandbox_gateway=$SANDBOX_GATEWAY \
    sandbox_fusion_url=$SANDBOX_FUSION_URL